# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Du-IN (Discrete units-guided mask modeling for decoding speech from Intracranial Neural signals) is a NeurIPS 2024 research implementation for speech decoding from sEEG (stereo-electroencephalography) recordings. The project implements a three-stage training pipeline (VQ-VAE → MAE → Classification) for Chinese word reading classification from brain signals.

## Environment Setup

Create and activate the conda environment:
```bash
conda env create -f environment-torch.yml
conda activate brain2vec-torch
```

## Training Pipeline

The Du-IN model follows a three-stage training process:

### Stage 1: Train VQ-VAE Model
```bash
cd ./train/duin
python run_vqvae.py --seed 42 --subjs 001
```
- Trains vector-quantized neural signal prediction
- Requires 1x NVIDIA Tesla V100 32GB or better
- Outputs checkpoint to `./pretrains/duin/{subj_id}/vqvae/model/checkpoint-399.pth`

### Stage 2: Train MAE Model
```bash
cd ./train/duin
python run_mae.py --seed 42 --subjs 001 \
    --vqkd_ckpt ./pretrains/duin/001/vqvae/model/checkpoint-399.pth
```
- Trains masked autoencoder by predicting neural codex indices from VQ-VAE
- Requires VQ-VAE checkpoint from Stage 1
- Outputs checkpoint to `./pretrains/duin/{subj_id}/mae/model/checkpoint-399.pth`

### Stage 3: Train Classification Model
```bash
cd ./train/duin
python run_cls.py --seeds 42 --subjs 001 --subj_idxs 0 \
    --pt_ckpt ./pretrains/duin/001/mae/model/checkpoint-399.pth
```
- Fine-tunes for 61-word classification task
- Can run with multiple seeds for statistical evaluation
- `--pt_ckpt` is optional; omit for training from scratch

### Batch Training
Use `run.sh` for batch training across all 12 subjects:
```bash
bash run.sh
```

## Architecture Overview

### Directory Structure

- **`models/`**: Model implementations for various architectures
  - `duin/`: Main Du-IN model (VQ-VAE, MAE, Classification, Alignment, LLM variants)
  - `brainbert/`, `brant/`, `neurobert/`, etc.: Baseline models for comparison
  - Each model directory contains `layers/` subdirectory with component implementations

- **`train/`**: Training scripts mirroring model structure
  - Each subdirectory corresponds to a model in `models/`
  - Contains `run_*.py` scripts for different training stages

- **`params/`**: Parameter configuration files
  - `*_params.py`: DotDict-based hierarchical parameter generation
  - Defines model architecture, training hyperparameters, and dataset-specific settings

- **`utils/`**: Utility modules
  - `DotDict.py`: Hierarchical parameter container
  - `Paths.py`: Path management for experiments
  - `data/seeg/`: Dataset loading and preprocessing
  - `model/torch/`: PyTorch-specific utilities (metrics, optimizers, distributed training)

- **`data/`**: Dataset symlink (download from HuggingFace)
- **`pretrains/`**: Pre-trained model checkpoints symlink

### Model Architecture Flow

**Du-IN VQ-VAE** (`models/duin/duin.py:duin_vqvae`):
```
Input (batch, n_channels, seq_len)
  → SubjectBlock: (batch, seq_len, n_channels) → (batch, seq_len, d_neural)
  → PatchTokenizer: (batch, seq_len, d_neural) → (batch, token_len, d_model)
  → TimeEmbedding + TransformerStack (Encoder)
  → LaBraMVectorQuantizer: Creates discrete neural codex
  → TransformerStack (Decoder)
  → TimeRGSHead: (batch, token_len, d_model) → (batch, seq_len, d_neural)
```

**Du-IN MAE** (`models/duin/duin.py:duin_mae`):
```
Input → SubjectBlock → PatchTokenizer
  → Masking layer
  → TransformerStack (Encoder)
  → Predicts codex indices from VQ-VAE neural codex
```

**Du-IN Classification** (`models/duin/duin.py:duin_cls`):
```
Input → SubjectBlock → PatchTokenizer
  → TransformerStack (Encoder)
  → ClsHead: (batch, token_len, d_model) → (batch, n_labels)
```

### Key Design Patterns

1. **Region-Level Tokenization**: Models use 1D depthwise convolution in SubjectBlock to fuse channels from specific brain regions (vSMC and STG) before tokenization, following neuroscience-inspired design.

2. **Parameter Hierarchy**: All models use `DotDict`-based params with `.iteration()` method for dynamic learning rate scheduling (warmup + cosine decay).

3. **Multi-Stage Pre-training**: Du-IN leverages discrete codex from VQ-VAE as supervision signal for MAE, enabling effective self-supervised learning on brain signals.

4. **Subject-Specific Training**: Each subject (001-012) requires separate model training due to individual differences in electrode placement and brain anatomy.

## Dataset Information

- **Source**: Chinese word-reading sEEG dataset from 12 subjects
- **Download**: Available at [HuggingFace](https://huggingface.co/datasets/liulab-repository/Du-IN)
- **Format**: Preprocessed sEEG signals with word labels
- **Size**: ~3 hours of recordings for 61-word classification
- **Brain Regions**: Focuses on language-related networks (vSMC and STG)

Place downloaded `data/` and `pretrains/` directories in project root (or create symlinks).

## Common Modifications

### Adding a New Model Variant

1. Create model class in `models/duin/duin.py` (e.g., `duin_new_variant`)
2. Add parameter generator in `params/duin_params.py` (e.g., `duin_new_variant_params`)
3. Create training script in `train/duin/` (e.g., `run_new_variant.py`)
4. Follow existing patterns for initialization, forward pass, and loss computation

### Changing Model Architecture

- Modify layer components in `models/duin/layers/`
- Update corresponding parameter generation in `params/duin_params.py`
- Ensure tensor shape compatibility throughout the pipeline

### Multi-GPU Training

The codebase supports distributed training:
- Training scripts check for `LOCAL_RANK` environment variable
- Use `torch.distributed` for multi-GPU setups
- Launch with `torchrun` or similar distributed launchers

## File Naming Conventions

- Subject IDs: Zero-padded 3 digits (001, 002, ..., 012)
- Checkpoints: `checkpoint-{epoch}.pth` (epoch 399 is typically the final checkpoint)
- Training scripts: `run_{stage}.py` where stage ∈ {vqvae, mae, cls, align}
