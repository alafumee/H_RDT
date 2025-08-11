#!/bin/bash

# H-RDT RobotWin2 Data Processing Pipeline
# This script runs the complete RobotWin2 data processing pipeline

# Setup environment (source the setup script if not already done)
if [ -z "$ROBOTWIN2_DATA_ROOT" ]; then
    source "$(dirname "$0")/setup_robotwin2.sh"
fi

# Change to project root
cd "$HRDT_PROJECT_ROOT"

# Define output paths
export STATS_OUTPUT_PATH="${HRDT_OUTPUT_DIR}/stats.json"
export OUTLIER_OUTPUT_PATH="${HRDT_OUTPUT_DIR}/outlier_files.txt"

echo "Starting RobotWin2 data processing pipeline..."
echo "Data Root: $ROBOTWIN2_DATA_ROOT"
echo "Output Dir: $HRDT_OUTPUT_DIR"

# Step 1: Calculate dataset statistics (Optional)
# echo "Step 1: Calculating dataset statistics..."
# python datasets/robotwin2/calc_stat.py \
#     --root_dir "$ROBOTWIN2_DATA_ROOT" \
#     --output_path "$STATS_OUTPUT_PATH" \
#     --outlier_path "$OUTLIER_OUTPUT_PATH" \
#     --num_processes "$NUM_PROCESSES"

# Step 2: Encode language embeddings
echo "Step 1: Encoding language embeddings..."
python datasets/robotwin2/encode_lang_batch.py

echo "RobotWin2 pipeline completed!"
echo "Generated files:"
echo "  - Statistics: $STATS_OUTPUT_PATH"
echo "  - Outlier files: $OUTLIER_OUTPUT_PATH"
echo "  - Language embeddings: *.pt files in task directories" 