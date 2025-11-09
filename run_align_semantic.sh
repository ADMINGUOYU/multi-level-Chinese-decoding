#!/bin/bash
################################################################################
# Du-IN Semantic Alignment Training Script
#
# This script trains the semantic embedding alignment model using pre-trained MAE.
# Adjust parameters below for hyperparameter tuning experiments.
################################################################################

# Experiment Target: try to train on sub11

# Get the absolute path of this script (MUST be before changing directory)
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Basic configuration
SEEDS="42"                    # Random seeds (space-separated for multiple runs)
# Subject list - will loop through all subjects sequentially
ALL_SUBJS=("001" "002" "003" "004" "005" "006" "007" "008" "009" "010" "011" "012")

# Learning rate schedule
LR_MIN=1e-5                   # Minimum learning rate (cosine annealing end)
LR_MAX=3e-4                   # Maximum learning rate (after warmup)

# Training schedule
N_EPOCHS=300                  # Total number of training epochs
WARMUP_EPOCHS=25              # Number of warmup epochs (linear warmup)
BATCH_SIZE=64                 # Batch size for training

# Loss scales (adjust to balance multi-task objectives)
CONTRA_LOSS_SCALE=0.5         # Contrastive loss scale (should be relatively small)
ALIGN_LOSS_SCALE=5.0          # Alignment loss scale (primary objective)

# Alignment head architecture
D_HIDDEN="2048,1024,768"           # Hidden layer dimensions (comma-separated, e.g., "1024,512" or "2048,1024,512")
ALIGN_DROPOUT=0.1             # Dropout rate for alignment head
D_OUTPUT=768                  # Output embedding dimension (should match target embeddings)

# Encoder dropout rates
ATTN_DROPOUT=0.2              # Attention dropout rate
FF_DROPOUT="0.2,0.0"          # Feedforward dropout rates (comma-separated: [layer1, layer2])

# Contrastive learning parameters
CONTRA_D_HIDDEN=32            # Contrastive projection dimension
CONTRA_LOSS_MODE="clip_orig"  # Contrastive loss mode: [clip, clip_orig, unicl]

# Encoder architecture
N_BLOCKS=8                    # Number of transformer blocks
N_HEADS=8                     # Number of attention heads

################################################################################
# Run training
################################################################################

# Change to training directory
cd train/duin

# Loop through all subjects sequentially
for SUBJ in "${ALL_SUBJS[@]}"; do
    echo "========================================="
    echo "Training semantic alignment for subject ${SUBJ}..."
    echo "========================================="

    # Set subject-specific pretrained checkpoint
    PT_CKPT="./pretrains/duin/${SUBJ}/mae/model/checkpoint-399.pth"

    python run_align_semantic.py \
        --seeds ${SEEDS} \
        --subjs ${SUBJ} \
        --subj_idxs 0 \
        --pt_ckpt ${PT_CKPT} \
        --lr_min ${LR_MIN} \
        --lr_max ${LR_MAX} \
        --n_epochs ${N_EPOCHS} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --contra_loss_scale ${CONTRA_LOSS_SCALE} \
        --align_loss_scale ${ALIGN_LOSS_SCALE} \
        --d_hidden ${D_HIDDEN} \
        --align_dropout ${ALIGN_DROPOUT} \
        --d_output ${D_OUTPUT} \
        --attn_dropout ${ATTN_DROPOUT} \
        --ff_dropout ${FF_DROPOUT} \
        --contra_d_hidden ${CONTRA_D_HIDDEN} \
        --contra_loss_mode ${CONTRA_LOSS_MODE} \
        --n_blocks ${N_BLOCKS} \
        --n_heads ${N_HEADS} \
        --run_script "${SCRIPT_PATH}"

    echo "Finished training subject ${SUBJ}"
    echo ""
done

echo "========================================="
echo "All subjects semantic alignment training completed!"
echo "========================================="

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

# Experiment 2: Deeper alignment head
# D_HIDDEN="2048,1024,512"

# Experiment 3: Stronger alignment loss
# ALIGN_LOSS_SCALE=10.0

# Experiment 4: Different contrastive loss mode
# CONTRA_LOSS_MODE="clip"
