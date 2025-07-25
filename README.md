# H-RDT: Human Manipulation Enhanced Bimanual Robotic Manipulation

H-RDT (**H**uman to **R**obotics **D**iffusion **T**ransformer) is a novel approach that leverages **large-scale egocentric human manipulation data** to enhance robot manipulation capabilities. Our key insight is that large-scale egocentric human manipulation videos with paired 3D hand pose annotations provide rich behavioral priors that capture natural manipulation strategies and can benefit robotic policy learning.

## ✨ Key Features

- **Human Data Pre-training**: Pre-trains on 829 hours of EgoDx egocentric human manipulation videos (338K episodes)
- **Cross-Embodiment Transfer**: Modular architecture enables effective knowledge transfer from human demonstrations to diverse robotic platforms
- **Flow Matching Training**: Uses flow matching for stable and efficient policy learning instead of traditional diffusion
- **48-Dimensional Action Space**: Unified 48D hand pose representation bridges human and robot embodiments
- **Two-Stage Training**: (1) Pre-train on large-scale human data, (2) Cross-embodiment fine-tuning on robot-specific data

## 🎯 Core Innovation

H-RDT addresses the fundamental challenge of **data scarcity** in robotic manipulation by:

1. **Leveraging Human Behavioral Priors**: Uses the vast repository of human manipulation videos with 3D hand pose annotations from EgoDx dataset
2. **Cross-Embodiment Knowledge Transfer**: Modular transformer architecture with specialized action encoders/decoders that enable effective transfer from human demonstrations to diverse robot platforms  
3. **Unified Action Representation**: 48-dimensional hand pose encoding captures essential bimanual manipulation information transferable across embodiments

## 🏗️ Method Overview

### 48-Dimensional Human Hand Action Representation
- **Bilateral wrist poses**: Position (3D) + orientation (6D) for both hands = 18 dimensions
- **Fingertip positions**: 3D coordinates for all fingers on both hands = 30 dimensions  
- **Total**: 48 dimensions capturing essential bimanual dexterous manipulation information

### Two-Stage Training Paradigm

**Stage 1: Human Data Pre-training**
- Train on complete EgoDx dataset (338K+ trajectories, 194 manipulation tasks)
- Learn general manipulation priors from human demonstrations
- Develop robust action representations for dexterous bimanual manipulation

**Stage 2: Cross-Embodiment Fine-tuning**  
- Transfer vision encoder, language encoder, and transformer backbone weights
- Re-initialize action adapters for target robot action space (e.g., 14D for dual 7-DOF arms)
- Preserve learned manipulation semantics while adapting to robot morphologies

## 🚀 Installation

1. **Create conda environment:**
   ```bash
   conda create -n hrdt python=3.10
   conda activate hrdt
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download pre-trained models:**
   ```bash
   export HF_ENDPOINT=https://hf-mirror.com
   huggingface-cli download --resume-download hongzhe2002/H-RDT --local-dir ./
   ```

## 🔧 Usage

### Stage 1: Human Data Pre-training (EgoDx)

#### Data Preprocessing
Before training, preprocess the EgoDx dataset:

1. **Configure paths:**
   ```bash
   # Edit datasets/pretrain/setup_pretrain.sh with your paths
   nano datasets/pretrain/setup_pretrain.sh
   
   # Set your EgoDx dataset and T5 model paths:
   export EGODEX_DATA_ROOT="/path/to/your/egodx/dataset"
   export T5_MODEL_PATH="/path/to/your/t5-v1_1-xxl"
   ```

2. **Setup environment:**
   ```bash
   source datasets/pretrain/setup_pretrain.sh
   ```

3. **Run data processing pipeline:**
   ```bash
   # Automatically runs: precompute_48d_actions.py → calc_stat.py → encode_lang_batch.py
   ./datasets/pretrain/run_pretrain_pipeline.sh
   ```

#### Start Pre-training
After data preprocessing is complete:

**1. EgoDx Pretrain (fresh start):**
1. Configure dataset:
   ```python
   # Edit datasets/dataset.py line ~45
   self.dataset_name = "egodx"
   ```
2. Run training:
   ```bash
   bash pretrain.sh
   ```

**2. Pretrain Resume:**
Edit `pretrain.sh`, add this line:
```bash
--resume_from_checkpoint="checkpoint-450000" \
```

### Stage 2: Cross-Embodiment Fine-tuning

#### Data Preprocessing (for RobotWin2)
Before fine-tuning, preprocess the robot dataset:

1. **Configure paths:**
   ```bash
   # Edit datasets/robotwin2/setup_robotwin2.sh with your paths
   nano datasets/robotwin2/setup_robotwin2.sh
   
   # Set your RobotWin2 dataset and T5 model paths:
   export ROBOTWIN2_DATA_ROOT="/path/to/your/robotwin2/dataset"
   export T5_MODEL_PATH="/path/to/your/t5-v1_1-xxl"
   ```

2. **Setup environment:**
   ```bash
   source datasets/robotwin2/setup_robotwin2.sh
   ```

3. **Run data processing pipeline:**
   ```bash
   # Automatically runs: calc_stat.py → encode_lang_batch.py
   ./datasets/robotwin2/run_robotwin2_pipeline.sh
   ```

**3. Robot Fine-tuning (load human pre-trained backbone):**
1. Configure dataset:
   ```python
   # Edit datasets/dataset.py line ~45
   self.dataset_name = "robotwin_agilex"  # or your robot name
   
   # Add your dataset initialization if not exists:
   elif self.dataset_name == "your_robot":
       self.hdf5_dataset = YourRobotDataset(config=config)
   ```
2. Run training:
   ```bash
   bash finetune.sh  # Already configured with pretrained_backbone_path
   ```

**4. Finetune Resume:**
Edit your current finetune script, make these changes:
```bash
# Change this line:
--mode="finetune" \
# To:
--mode="pretrain" \

# And add:
--resume_from_checkpoint="checkpoint-5000" \
```

## 🎯 Training Modes

| Training Scenario | Base Script | Required Shell Script Modifications | Mode & Key Parameters |
|-------------------|-------------|-------------------------------------|----------------------|
| **Human Pretrain (Fresh)** | `pretrain.sh` | `--mode="pretrain"` | Start pretraining on EgoDx human data |
| **Human Pretrain Resume** | `pretrain.sh` | Add: `--resume_from_checkpoint="checkpoint-450000" \` | `--mode="pretrain"` |
| **Robot Fine-tuning** | `finetune.sh` | Change: `--mode="finetune" \`<br>Add: `--pretrained_backbone_path="./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin" \`<br>Change: `--config_path="configs/hrdt_finetune.yaml" \` | Load human pre-trained backbone, fresh action layers |
| **Robot Finetune Resume** | Your finetune script | Change: `--mode="finetune"` → `--mode="pretrain"`<br>Add: `--resume_from_checkpoint="checkpoint-5000" \` | Continue robot fine-tuning |

### Dataset Configuration

Before training, you need to configure the dataset in `datasets/dataset.py`:

#### For Human Pre-training (EgoDx):
```python
# In datasets/dataset.py, line ~45
self.dataset_name = "egodx"

# The EgoDxDataset will be automatically initialized
```

#### For Robot Fine-tuning:
```python
# In datasets/dataset.py, line ~45  
self.dataset_name = "your_robot_name"  # e.g., "robotwin_agilex"

# Add your dataset to the initialization logic:
elif self.dataset_name == "your_robot_name":
    self.hdf5_dataset = YourRobotDataset(
        config=config,
        # your dataset parameters
    )
```

#### Adding New Robot Datasets:
1. Create your dataset folder: `datasets/your_robot/`
2. Implement your dataset class (see `datasets/robotwin2/` as example)
3. Create data processing scripts (see `datasets/pretrain/` or `datasets/robotwin2/` as examples)
4. Import in `datasets/dataset.py`
5. Add initialization logic in `VLAConsumerDataset.__init__`

### Key Configuration Files
- `configs/hrdt_pretrain.yaml`: Human pre-training configuration
- `configs/hrdt_finetune.yaml`: Robot fine-tuning configuration  
- `datasets/dataset.py`: Dataset selection and initialization
- Modify `state_dim`, `action_dim`, `output_size` for your robot

### Path Support
✅ **Relative paths supported**: `./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin`  
✅ **Absolute paths supported**: `/full/path/to/checkpoint/pytorch_model.bin`

## 📊 Model Architecture

H-RDT consists of:
- **Vision Encoder**: DinoV2 + SigLIP-based image feature extraction
- **Language Encoder**: T5-XXL-based instruction encoding  
- **Modular Action Encoder**: Handles 48D human poses (pre-train) and robot actions (fine-tune)
- **Transformer Backbone**: 16-layer transformer with self-attention and cross-attention (~2B parameters)
- **Modular Action Decoder**: Outputs action sequences using flow matching
- **Flow Matching**: Replaces traditional diffusion for stable and efficient training

## 🛠️ Project Structure

```
h_rdt/
├── models/
│   ├── rdt_runner.py          # Main model implementation  
│   └── encoder/               # Vision and language encoders
├── train/
│   ├── train.py              # Training script
│   └── sample.py             # Sampling/evaluation
├── configs/
│   ├── hrdt_pretrain.yaml    # Human pre-training config
│   └── hrdt_finetune.yaml    # Robot fine-tuning config
├── datasets/
│   ├── pretrain/             # EgoDx human data processing
│   └── robotwin2/            # Robot data processing
├── pretrain.sh              # Human pre-training script
└── finetune.sh              # Robot fine-tuning script
```