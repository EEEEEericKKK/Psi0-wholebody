#!/bin/bash

# H-RDT Pretrain Data Processing Setup
# Set your paths here

# Required paths - modify these according to your environment
export EGODEX_DATA_ROOT="/lustre/fs12/portfolios/nvr/projects/nvr_av_foundations/datasets/ego-dex"
# export EGODEX_DATA_ROOT="/path/to/raw/ego-dex"
export T5_MODEL_PATH="/hfm/cache/checkpoints/t5-v1_1-xxl"

# Project structure (auto-detected)
export HRDT_PROJECT_ROOT="$(pwd)/src/h_rdt"
export HRDT_CONFIG_PATH="${HRDT_PROJECT_ROOT}/configs/hrdt_pretrain.yaml"
export HRDT_OUTPUT_DIR="${HRDT_PROJECT_ROOT}/datasets/pretrain"

echo "HRDT_PROJECT_ROOT: $HRDT_PROJECT_ROOT"

# Processing parameters
export NUM_PROCESSES=64
export FORCE_OVERWRITE=false

# Create output directory
mkdir -p "$HRDT_OUTPUT_DIR"

# Add project to Python path
export PYTHONPATH="${HRDT_PROJECT_ROOT}:${PYTHONPATH}"


echo "Pretrain environment setup completed"
echo "Data Root: $EGODEX_DATA_ROOT"
echo "T5 Model: $T5_MODEL_PATH"
echo "Output Dir: $HRDT_OUTPUT_DIR" 
