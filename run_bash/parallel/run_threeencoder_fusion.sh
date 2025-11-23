#!/bin/bash
################################################################################
# Three-Encoder Fusion Classifier Training Launcher
#
# Minimal wrapper to launch train/duin/run_threeencoder_fusion.py across subjects
# and seeds. Mirrors the style of other run_bash/parallel scripts in the repo.
#
# Edit the checkpoint paths and settings below to match your environment.
################################################################################

SCRIPT_PATH="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# General experiment settings
SEEDS="42"
ALL_SUBJS=("002")
LR_MIN=1e-5
LR_MAX=2e-4
N_EPOCHS=400
WARMUP_EPOCHS=20
BATCH_SIZE=32

# Fusion head config
FUSION_D_HIDDEN="1024,512,256"
FUSION_DROPOUT=0.4
CLS_LOSS_SCALE=1.0

# Other flags
FREEZE_ENCODERS=true
FREEZE_TASK_HEADS=false
RUN_SCRIPT="${SCRIPT_PATH}"

################################################################################
# Multi-Task Learning Parameters
################################################################################

# Task weights for loss combination (only used when use_uncertainty_weighting=false)
TASK_WEIGHT_SEMANTIC=1.0      # Weight for semantic task
TASK_WEIGHT_VISUAL=1.0        # Weight for visual task
TASK_WEIGHT_ACOUSTIC=1.0      # Weight for acoustic task

# Multi-task learning strategy flags
USE_UNCERTAINTY_WEIGHTING="--use_uncertainty_weighting"  # Set to "--use_uncertainty_weighting" to enable auto-balancing, default as ""
# USE_UNCERTAINTY_WEIGHTING=""  # Set to "--use_uncertainty_weighting" to enable auto-balancing, default as ""
ACOUSTIC_USE_CONTRA=""        # Set to "--acoustic_use_contra" to add contrastive loss to acoustic task, default as ""

# Move to train/duin directory
cd "$(dirname "${BASH_SOURCE[0]}")/train/duin" || exit 1

for SUBJ in "${ALL_SUBJS[@]}"; do
    echo "================================================================"
    echo "Training three-encoder fusion classifier for subject ${SUBJ}..."
    echo "================================================================"

    # Checkpoints (update to point to your trained checkpoint files)
    # PT_MULTITASK_CKPT="${PROJECT_ROOT}"
    SEMANTIC_CKPT="${PROJECT_ROOT}/sample/002/semantic/train/ckpt/best_epoch_098_loss_2.60096.pt"
    VISUAL_CKPT="${PROJECT_ROOT}/sample/002/visual/train/ckpt/best_epoch_062_loss_2.02765.pt"
    ACOUSTIC_CKPT="${PROJECT_ROOT}/sample/002/acoustic/train/ckpt/best_epoch_182_tone2_acc_0.7014.pt"

    python run_threeencoder_fusion.py \
        --seeds ${SEEDS} \
        --subjs ${SUBJ} \
        --subj_idxs 0 \
        --semantic_ckpt "${SEMANTIC_CKPT}" \
        --visual_ckpt "${VISUAL_CKPT}" \
        --acoustic_ckpt "${ACOUSTIC_CKPT}" \
        --n_epochs ${N_EPOCHS} \
        --warmup_epochs ${WARMUP_EPOCHS} \
        --batch_size ${BATCH_SIZE} \
        --lr_min ${LR_MIN} \
        --lr_max ${LR_MAX} \
        --fusion_d_hidden "${FUSION_D_HIDDEN}" \
        --fusion_dropout ${FUSION_DROPOUT} \
        --cls_loss_scale ${CLS_LOSS_SCALE} \
        $( [ "${FREEZE_ENCODERS}" = true ] && echo "--freeze_encoder" ) \
        $( [ "${FREEZE_TASK_HEADS}" = true ] && echo "--freeze_task_heads" ) \
        --run_script "${RUN_SCRIPT}" \
        --task_weight_semantic ${TASK_WEIGHT_SEMANTIC} \
        --task_weight_visual ${TASK_WEIGHT_VISUAL} \
        --task_weight_acoustic ${TASK_WEIGHT_ACOUSTIC} \
        # ${USE_UNCERTAINTY_WEIGHTING} \
        ${ACOUSTIC_USE_CONTRA} \

    echo "==== Finished subject ${SUBJ} ===="
done

echo "================================="
echo "All subjects training completed!"
echo "================================="