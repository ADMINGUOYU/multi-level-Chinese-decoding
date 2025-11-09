#!/bin/bash
################################################################################
# Du-IN Acoustic Tone Classification Training Script
#
# This script trains the acoustic tone classification model for predicting
# Chinese tones (tone1 and tone2) from brain signals.
# Adjust parameters below for hyperparameter tuning experiments.
#
# OVERFITTING PREVENTION STRATEGY:
# - Increased dropout rates in encoder (ATTN_DROPOUT=0.3, FF_DROPOUT="0.3,0.2")
# - High dropout in classification head (CLS_DROPOUT=0.5)
# - Loss normalization applied to ensure stable gradients
# - These settings help model generalize better to unseen data
################################################################################

# Experiment Target: test

# Get the absolute path of this script (MUST be before changing directory)
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Basic configuration
SEEDS="42"                    # Random seeds (space-separated for multiple runs)
SUBJS="001"                   # Subject IDs (space-separated for multiple subjects)
SUBJ_IDXS="0"                 # Subject indices (must match number of subjects)
PT_CKPT="./pretrains/duin/001/mae/model/checkpoint-399.pth"  # Pre-trained checkpoint path

# Learning rate schedule
LR_MIN=1e-5                   # Minimum learning rate (cosine annealing end)
LR_MAX=5e-4                   # Maximum learning rate (after warmup)

# Training schedule
N_EPOCHS=200                  # Total number of training epochs
WARMUP_EPOCHS=20              # Number of warmup epochs (linear warmup)
BATCH_SIZE=64                 # Batch size for training

# Encoder dropout rates (increased to prevent overfitting)
# Higher dropout rates help model generalize better by preventing co-adaptation of features
ATTN_DROPOUT=0.3              # Attention dropout rate (increased from 0.1 to 0.3)
FF_DROPOUT="0.3,0.2"          # Feedforward dropout rates (increased from "0.1,0.0" to "0.3,0.2")
                              # Format: "layer1,layer2" - applies different dropout to each FF sublayer

# Encoder architecture
N_BLOCKS=8                    # Number of transformer blocks
N_HEADS=8                     # Number of attention heads

# Classification head architecture
D_HIDDEN="128,128"            # Hidden layer dimensions (comma-separated, e.g., "512,256" or empty for no hidden layers)
                              # Multi-layer head allows for more complex decision boundaries
CLS_DROPOUT=0.5               # Dropout rate for classification head (increased from 0.2 to 0.5)
                              # Higher dropout in classification head is critical for preventing overfitting
                              # This is typically the layer most prone to memorizing training data
USE_L2_NORM=true              # Apply L2 normalization to logits before computing loss
                              # This constrains prediction magnitudes and helps prevent overfitting
                              # Set to 'false' to disable (not recommended for overfitting scenarios)

################################################################################
# Run training
################################################################################

# Change to training directory
cd train/duin

python run_cls_acoustic.py \
    --seeds ${SEEDS} \
    --subjs ${SUBJS} \
    --subj_idxs ${SUBJ_IDXS} \
    --pt_ckpt ${PT_CKPT} \
    --lr_min ${LR_MIN} \
    --lr_max ${LR_MAX} \
    --n_epochs ${N_EPOCHS} \
    --warmup_epochs ${WARMUP_EPOCHS} \
    --batch_size ${BATCH_SIZE} \
    --attn_dropout ${ATTN_DROPOUT} \
    --ff_dropout ${FF_DROPOUT} \
    --n_blocks ${N_BLOCKS} \
    --n_heads ${N_HEADS} \
    --d_hidden ${D_HIDDEN} \
    --cls_dropout ${CLS_DROPOUT} \
    --use_l2_norm ${USE_L2_NORM} \
    --run_script "${SCRIPT_PATH}"

################################################################################
# Example: Multi-seed training for statistical evaluation
################################################################################
# Uncomment below to run with multiple seeds:
# SEEDS="42 123 456 789 1024"

################################################################################
# Example: Hyperparameter tuning experiments
################################################################################
# Experiment 1: Higher learning rate
# LR_MIN=1e-5 LR_MAX=1e-3

# Experiment 2: Larger batch size
# BATCH_SIZE=128

# Experiment 3: Deeper model
# N_BLOCKS=12

# Experiment 4: Different dropout rates
# ATTN_DROPOUT=0.2
# FF_DROPOUT="0.2,0.1"
# CLS_DROPOUT=0.3

# Experiment 5: Multi-layer classification head
# D_HIDDEN="512,256"
# CLS_DROPOUT=0.3

################################################################################
# Example: Multi-subject training
################################################################################
# Train on subject 001:
# SUBJS="001"
# SUBJ_IDXS="0"
# PT_CKPT="./pretrains/duin/001/mae/model/checkpoint-399.pth"
#
# Train on subjects 001 and 002 together:
# SUBJS="001 002"
# SUBJ_IDXS="0 1"
# PT_CKPT="./pretrains/duin/001/mae/model/checkpoint-399.pth"

################################################################################
# Training Details
################################################################################
# The model predicts two acoustic tones per Chinese word:
#   - tone1: First character's tone (5 classes: 1, 2, 3, 4, neutral)
#   - tone2: Second character's tone (5 classes: 1, 2, 3, 4, neutral)
#
# Training logs show accuracy for both tone1 and tone2 separately:
#   - Accuracy(train-tone1): [subject1%, subject2%, ...]
#   - Accuracy(train-tone2): [subject1%, subject2%, ...]
#   - Accuracy(validation-tone1): [subject1%, subject2%, ...]
#   - Accuracy(validation-tone2): [subject1%, subject2%, ...]
#   - Accuracy(test-tone1): [subject1%, subject2%, ...]
#   - Accuracy(test-tone2): [subject1%, subject2%, ...]
#
# Best epoch is selected based on average validation accuracy (tone1 + tone2)

################################################################################
# Hardware Requirements
################################################################################
# - Recommended: 1x NVIDIA Tesla V100 32GB or better
# - Training time: ~1-2 hours per subject (200 epochs)
# - GPU memory: ~8-12 GB depending on batch size
