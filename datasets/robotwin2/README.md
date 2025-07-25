# H-RDT RobotWin2 Data Processing

This directory contains scripts for processing RobotWin2 dataset for H-RDT fine-tuning.

## Overview

The RobotWin2 data processing consists of two steps:
1. **Calculate Statistics** - Compute min/max values for action normalization
2. **Encode Language** - Generate T5 embeddings for task instructions

## Quick Start

### 1. Setup Environment

Edit the paths in `setup_robotwin2.sh` according to your environment:

```bash
# Edit the script with your paths
nano datasets/robotwin2/setup_robotwin2.sh

# Then source it to set up environment variables
source datasets/robotwin2/setup_robotwin2.sh
```

**Required paths to configure:**
- `ROBOTWIN2_DATA_ROOT`: Path to your RobotWin2 dataset
- `T5_MODEL_PATH`: Path to your T5-v1_1-xxl model

### 2. Run Complete Pipeline

```bash
# Run both steps automatically
./datasets/robotwin2/run_robotwin2_pipeline.sh
```

### 3. Run Individual Steps (Optional)

```bash
# Step 1: Calculate statistics
python datasets/robotwin2/calc_stat.py

# Step 2: Encode language embeddings
python datasets/robotwin2/encode_lang_batch.py
```

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `ROBOTWIN2_DATA_ROOT` | RobotWin2 dataset root directory | Required |
| `T5_MODEL_PATH` | T5-v1_1-xxl model path | Required |
| `HRDT_PROJECT_ROOT` | H-RDT project root | Auto-detected |
| `HRDT_CONFIG_PATH` | Config file path | `configs/hrdt_finetune.yaml` |
| `HRDT_OUTPUT_DIR` | Output directory | `datasets/robotwin2` |
| `NUM_PROCESSES` | Number of processes | 8 |
| `NUM_GPUS` | Number of GPUs | 8 |
| `PROCESSES_PER_GPU` | Processes per GPU | 4 |

### Command Line Arguments

Scripts accept command line arguments that override environment variables:

```bash
# Calculate statistics with custom settings
python datasets/robotwin2/calc_stat.py

# Encode language embeddings (uses environment variables)  
python datasets/robotwin2/encode_lang_batch.py
```

## Directory Structure

```
datasets/robotwin2/
├── setup_robotwin2.sh              # Environment setup script
├── run_robotwin2_pipeline.sh       # Complete pipeline runner
├── calc_stat.py                    # Step 1: Calculate statistics
├── encode_lang_batch.py            # Step 2: Encode language
├── robotwin_agilex_dataset.py      # RobotWin2 dataset loader
├── stats.json                     # Generated statistics file
├── task_instructions.csv          # Task instruction mapping
└── README.md                      # This file
```

## Expected Dataset Structure

Your RobotWin2 dataset should be organized as:

```
$ROBOTWIN2_DATA_ROOT/
├── task1/
│   ├── demo_clean/data/
│   │   ├── episode0.hdf5
│   │   ├── episode1.hdf5
│   │   └── ...
│   └── task1.pt               # Generated T5 embeddings
├── task2/
├── task3/
└── ...
```

## Output Files

After processing, you'll have:

1. **Statistics**: `stats.json` with min/max values for action normalization
2. **Language Embeddings**: `task_name.pt` files in each task directory
3. **Log Files**: Processing logs and status information