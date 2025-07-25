# H-RDT: Hierarchical Robotics Diffusion Transformer

H-RDT (Hierarchical Robotics Diffusion Transformer) is an advanced **imitation learning** model based on **Diffusion Transformer** architecture, designed for **multi-modal robot manipulation** tasks. Given language instructions and RGB images, H-RDT can predict future robot actions with superior performance and generalizability.

## ✨ Key Features

- **Video-Free Training**: Optimized for image+language conditioning without video dependencies
- **Hierarchical Architecture**: Advanced transformer blocks with cross-attention mechanisms
- **Multi-Modal Support**: Language instructions + RGB images from multiple camera views
- **Robot Agnostic**: Compatible with various robot embodiments and action spaces

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

### Pre-training

#### Data Preprocessing
Before training, preprocess the EgoDex dataset:

1. **Configure paths:**
   ```bash
   # Edit datasets/pretrain/setup_pretrain.sh with your paths
   nano datasets/pretrain/setup_pretrain.sh
   
   # Set your EgoDex dataset and T5 model paths:
   export EGODEX_DATA_ROOT="/path/to/your/egodex/dataset"
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

#### Start Training
After data preprocessing is complete:

**1. EgoDx Pretrain (fresh start):**
1. Configure dataset:
   ```python
   # Edit datasets/dataset.py line ~45
   self.dataset_name = "egodex"
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

### Fine-tuning

#### Data Preprocessing (for RobotWin2)
Before fine-tuning, preprocess the RobotWin2 dataset:

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

**3. New Robot Finetune (load pretrained backbone):**
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
| **Pretrain (Fresh)** | Train from scratch | `--mode="pretrain"` | Start pretraining on EgoDex |
| **Pretrain Resume** | `pretrain.sh` | Add: `--resume_from_checkpoint="checkpoint-450000" \` | `--mode="pretrain"` |
| **New Robot Finetune** | `pretrain.sh` or new script | Change: `--mode="finetune" \`<br>Add: `--pretrained_backbone_path="./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin" \`<br>Change: `--config_path="configs/hrdt_finetune.yaml" \` | Load backbone, fresh action layers |
| **Finetune Resume** | Your finetune script | Change: `--mode="finetune"` → `--mode="pretrain"`<br>Add: `--resume_from_checkpoint="checkpoint-5000" \` | Continue finetune training |

### Dataset Configuration

Before training, you need to configure the dataset in `datasets/dataset.py`:

#### For Pretrain (EgoDex):
```python
# In datasets/dataset.py, line ~45
self.dataset_name = "egodex"

# The EgoDexDataset will be automatically initialized
```

#### For Finetune (Your Robot):
```python
# In datasets/dataset.py, line ~45  
self.dataset_name = "your_robot_name"  # e.g., "franka_kitchen"

# Add your dataset to the initialization logic:
elif self.dataset_name == "your_robot_name":
    self.hdf5_dataset = YourRobotDataset(
        config=config,
        # your dataset parameters
    )
```

#### Adding New Datasets:
1. Create your dataset folder: `datasets/your_robot/`
2. Implement your dataset class (see `datasets/robotwin2/` as example)
3. Create data processing scripts (see `datasets/pretrain/` or `datasets/robotwin2/` as examples)
4. Import in `datasets/dataset.py`
5. Add initialization logic in `VLAConsumerDataset.__init__`

### Key Configuration Files
- `configs/hrdt_pretrain.yaml`: Pre-training configuration
- `configs/hrdt_finetune.yaml`: Fine-tuning configuration  
- `datasets/dataset.py`: Dataset selection and initialization
- Modify `state_dim`, `action_dim`, `output_size` for your robot

### Path Support
✅ **Relative paths supported**: `./checkpoints/pretrain-0618/checkpoint-500000/pytorch_model.bin`  
✅ **Absolute paths supported**: `/full/path/to/checkpoint/pytorch_model.bin`

## 📊 Model Architecture

H-RDT consists of:
- **Vision Encoder**: SigLIP-based image feature extraction
- **Language Encoder**: T5-based instruction encoding  
- **H-RDT Transformer**: Hierarchical diffusion transformer blocks
- **Action Decoder**: Multi-step action sequence prediction

## 🛠️ Project Structure

```
h_rdt/
├── models/
│   ├── hrdt_runner.py          # Main model implementation
│   └── hrdt/                   # Core model architecture
├── train/
│   ├── train.py               # Training script
│   └── sample.py              # Sampling/evaluation
├── configs/
│   ├── hrdt_pretrain.yaml     # Pre-training config
│   └── hrdt_finetune.yaml     # Fine-tuning config
├── datasets/                  # Dataset loaders
├── pretrain.sh               # Pre-training script
└── finetune.sh              # Fine-tuning script
```

## 📝 Notes

- **Pre-training**: Based on EgoDex dataset with automated preprocessing pipeline
  - See `datasets/pretrain/README.md` for detailed data processing instructions
  - Use `datasets/pretrain/setup_pretrain.sh` to configure paths for different environments
- **Fine-tuning**: Example with RobotWin2.0, see `finetune.sh`
- **Configuration**: Modify YAML files for different robot embodiments
- **Models**: bak/ and checkpoints/ folders will be downloaded to project root
- **Cross-Platform**: Environment variable-based configuration works across different machines

## 🤝 Acknowledgments

Built upon the foundation of [RDT](https://github.com/thu-ml/RoboticsDiffusionTransformer)
