#!/bin/bash
################################################################################
# Du-IN Fusion Classifier Training Script
#
# This script trains the fusion classifier for 61-word classification using
# pretrained multi-task model (semantic + visual + acoustic).
# Adjust parameters below for hyperparameter tuning experiments.
################################################################################

# Experiment Target: run in all subjects

# Get the absolute path of this script (MUST be before changing directory)
SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# Basic configuration
SEEDS="42"                    # Random seeds (space-separated for multiple runs)
# Subject list - will loop through all subjects sequentially
# ALL_SUBJS=("001" "002" "003" "004" "005" "006" "007" "008" "009" "010" "011" "012")
ALL_SUBJS=("001")

# Learning rate schedule (lower LR for fine-tuning)
LR_MIN=1e-5                   # Minimum learning rate (cosine annealing end)
LR_MAX=1e-4                   # Maximum learning rate (after warmup)

# Training schedule
N_EPOCHS=250                  # Total number of training epochs
WARMUP_EPOCHS=20              # Number of warmup epochs (linear warmup)
BATCH_SIZE=32                 # Batch size for training

################################################################################
# Freezing Strategy
################################################################################

# Freeze encoder (SubjectBlock + Tokenizer + Encoder)
# Set to "--no_freeze_encoder" to enable end-to-end fine-tuning
FREEZE_ENCODER="--no_freeze_encoder"  # Default: freeze encoder for fine-tuning
# FREEZE_ENCODER="--freeze_encoder"  # Default: freeze encoder for fine-tuning

# Freeze task heads (semantic_head + visual_head + acoustic_heads)
# Set to "--freeze_task_heads" to freeze task heads
FREEZE_TASK_HEADS=""          # Default: do NOT freeze task heads (allow fine-tuning)

################################################################################
# Fusion Head Architecture
################################################################################

# Fusion head hidden dimensions (comma-separated)
# Input dimension: 768 + 768 + d_acoustic (automatically calculated)
# Default: 1664 → 512 → 256 → 61
FUSION_D_HIDDEN="1024,512,256"     # Hidden layer dimensions

# Dropout rate for fusion head
FUSION_DROPOUT=0.4            # Dropout probability

################################################################################
# Loss Configuration
################################################################################

# Classification loss scale
CLS_LOSS_SCALE=1.0            # Scale factor for classification loss

################################################################################
# Run training
################################################################################

# Change to training directory
cd train/duin

# Workaround for protobuf version incompatibility
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# Loop through all subjects sequentially
for SUBJ in "${ALL_SUBJS[@]}"; do
    echo "========================================="
    echo "Training fusion classifier for subject ${SUBJ}..."
    echo "========================================="

    # Set subject-specific pretrained multi-task checkpoint
    # IMPORTANT: This should point to the trained multitask model from Stage 1
    # PT_MULTITASK_CKPT="./pretrains/duin/${SUBJ}/multitask/model/checkpoint-299.pth"
    
    ##************此处现在DEBUG状态，用于测试，后续需要修改，存储最好的stage1 ckpt************##
    #certain weight，100epoch loss最低，后面loss就飘了
    # PT_MULTITASK_CKPT="/mnt/afs/250010218/multi-level-Chinese-decoding/summaries/2025-11-18/0/train/ckpt/checkpoint-099.pth"
    #unsertain weight
    PT_MULTITASK_CKPT="/mnt/afs/250010218/multi-level-Chinese-decoding/summaries/2025-11-17/1/train/ckpt/checkpoint-299.pth"
    #**************************************************************************##
   
    # Check if checkpoint exists
    if [ ! -f "${PT_MULTITASK_CKPT}" ]; then
        echo "ERROR: Multi-task checkpoint not found: ${PT_MULTITASK_CKPT}"
        echo "Please train the multi-task model first using run_multitask.sh"
        continue
    fi

    python run_fusion_cls.py \
        --seeds ${SEEDS} \
        --subjs ${SUBJ} \
        --subj_idxs 0 \
        --pt_multitask_ckpt ${PT_MULTITASK_CKPT} \
        --lr_min ${LR_MIN} \
        --lr_max ${LR_MAX} \
        --n_epochs ${N_EPOCHS} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        ${FREEZE_ENCODER} \
        ${FREEZE_TASK_HEADS} \
        --fusion_d_hidden ${FUSION_D_HIDDEN} \
        --fusion_dropout ${FUSION_DROPOUT} \
        --cls_loss_scale ${CLS_LOSS_SCALE} \
        --run_script "${SCRIPT_PATH}"

    echo "Finished training subject ${SUBJ}"
    echo ""
done

echo "========================================="
echo "All subjects fusion classifier training completed!"
echo "========================================="

################################################################################
# Example: Multi-seed training for statistical evaluation
################################################################################
# Uncomment below to run with multiple seeds:
# SEEDS="42 123 456 789 1024"

################################################################################
# Example: Enable end-to-end fine-tuning
################################################################################
# Uncomment below to enable end-to-end fine-tuning of all parameters:
# FREEZE_ENCODER="--no_freeze_encoder"
# FREEZE_TASK_HEADS=""

################################################################################
# Example: Freeze both encoder and task heads (only train fusion head)
################################################################################
# Uncomment below to freeze both encoder and task heads:
# FREEZE_ENCODER="--freeze_encoder"
# FREEZE_TASK_HEADS="--freeze_task_heads"

################################################################################
# Example: Experiment with different fusion head architectures
################################################################################
# Experiment 1: Deeper fusion head
# FUSION_D_HIDDEN="1024,512,256"

# Experiment 2: Shallower fusion head
# FUSION_D_HIDDEN="256"

# Experiment 3: Very deep fusion head with higher dropout
# FUSION_D_HIDDEN="1024,512,256,128"
# FUSION_DROPOUT=0.5

################################################################################
# Example: Use different multi-task checkpoint epoch
################################################################################
# If you want to use a different epoch checkpoint (e.g., epoch 249):
# PT_MULTITASK_CKPT="./pretrains/duin/${SUBJ}/multitask/model/checkpoint-249.pth"
