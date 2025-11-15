#!/usr/bin/env python3
"""
Fusion classifier training script for Du-IN model.
Uses pretrained multi-task model (semantic + visual + acoustic) for 61-word classification.

Created for two-stage multi-task learning implementation
@author: Claude Code
"""
import matplotlib.pyplot as plt
import torch
import os, time
import argparse
import copy as cp
import numpy as np
import scipy as sp
from collections import Counter
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
import utils; import utils.model.torch; import utils.data.seeg
from utils.data import load_pickle
from models.duin import duin_fusion_cls as duin_model

# GPU DEBUGGING: Disable cuDNN to test if it causes GPU training failure
import torch.backends.cudnn as cudnn
cudnn.enabled = False
cudnn.benchmark = False
cudnn.deterministic = True
print("WARNING: cuDNN DISABLED for debugging GPU training issue")

__all__ = [
    "init",
    "train",
]

# Global variables.
params = None; paths = None
model = None; optimizer = None

"""
init funcs
"""
# def init func
def init(params_):
    """
    Initialize `duin_fusion_cls` training variables.

    Args:
        params_: DotDict - The parameters of current training process.

    Returns:
        None
    """
    global params, paths
    # Initialize params.
    params = cp.deepcopy(params_)
    paths = utils.Paths(base=params.train.base, params=params)
    paths.run.logger.tensorboard = SummaryWriter(paths.run.train)
    # Initialize model.
    _init_model()
    # Initialize training process.
    _init_train()
    # Log the completion of initialization.
    msg = (
        "INFO: Complete the initialization of the training process with params ({})."
    ).format(params); print(msg); paths.run.logger.summaries.info(msg)

    # --- Add missing ckpt folder ---
    paths.run.ckpt = os.path.join(paths.run.train, "ckpt")
    os.makedirs(paths.run.ckpt, exist_ok=True)
    print(f"[INFO] Checkpoint directory created at: {paths.run.ckpt}")

    # --- Save training metadata file ---
    training_info_path = os.path.join(paths.run.train, "training_info.txt")
    subj_i = params.train.subjs[0]
    with open(training_info_path, "w") as f:
        f.write(f"Training Type: fusion_cls (61-word classification from fused embeddings)\n")
        f.write(f"Subject Number: {subj_i}\n")
        f.write(f"Multi-task Checkpoint: {params.train.pt_multitask_ckpt}\n")
        f.write(f"Freeze Encoder: {params.model.freeze_encoder}\n")
        f.write(f"Freeze Task Heads: {params.model.freeze_task_heads}\n")
        f.write(f"Fusion Head Architecture: {params.model.fusion.d_hidden}\n")
    print(f"[INFO] Training metadata saved to: {training_info_path}")

    # --- Save shell script if provided ---
    if hasattr(params.train, 'run_script') and params.train.run_script is not None:
        import shutil
        script_name = os.path.basename(params.train.run_script)
        script_dest = os.path.join(paths.run.script, script_name)
        try:
            shutil.copy2(params.train.run_script, script_dest)
            msg = f"[INFO] Shell script saved to: {script_dest}"
            print(msg); paths.run.logger.summaries.info(msg)
        except Exception as e:
            msg = f"[WARNING] Failed to save shell script: {e}"
            print(msg); paths.run.logger.summaries.warning(msg)

# def _init_model func
def _init_model():
    """
    Initialize model used in the training process.

    Args:
        None

    Returns:
        None
    """
    global model
    # Initialize model.
    model = duin_model(params=params.model)
    # Load pretrained multi-task weights
    if params.train.pt_multitask_ckpt is not None and os.path.exists(params.train.pt_multitask_ckpt):
        model.load_weight(path_ckpt=params.train.pt_multitask_ckpt)
    else:
        raise ValueError(f"Multi-task checkpoint not found: {params.train.pt_multitask_ckpt}")
    # Transfer model to device(s).
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    model = model.to(device=params.model.device)
    # Log the summary of model.
    msg = (
        "INFO: Model architecture summary:\n{}"
    ).format(model); print(msg); paths.run.logger.summaries.info(msg)

# def _init_train func
def _init_train():
    """
    Initialize training process.

    Args:
        None

    Returns:
        None
    """
    global optimizer
    # Initialize the optimizer of model.
    # Only optimize parameters that require gradients (respect freezing)
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=params.train.lr_i
    )

"""
train funcs
"""
# def train func
def train():
    """
    Train `duin_fusion_cls` for one epoch.

    Args:
        None

    Returns:
        None
    """
    # Initialize epochs_idx.
    epochs_idx = np.arange(params.train.epoch_start, params.train.n_epochs)
    # Execute train loop.
    for epoch_idx in epochs_idx:
        # Update params.train.epoch according to `epoch_idx`.
        params.train.epoch = epoch_idx
        # Log the start of an epoch.
        msg = (
            "INFO: Training process at epoch ({}) starts with lr ({:.6e})."
        ).format(params.train.epoch, params.train.lr_i)
        print(msg); paths.run.logger.summaries.info(msg)
        # We will use `_train_epoch` to finish the train process of one epoch.
        _train_epoch()
        # We will use `_valid_epoch` to validate the performance of one epoch.
        _valid_epoch()
        # We will use `_log_epoch` to log the information of one epoch.
        _log_epoch()
        # Update params after each epoch.
        params.iteration(params.train.epoch + 1)

# def _train_epoch func
def _train_epoch():
    """
    Train model for one epoch.

    Args:
        None

    Returns:
        None
    """
    global model, optimizer
    # Initialize `model.train`.
    model.train()
    # Initialize the record of time.
    time_start = time.time()
    # Get the number of subjects used for training.
    n_subjects = len(params.train.subjs)
    # Initialize training metrics
    train_loss = 0.; train_correct = 0; train_total = 0
    # Execute train loop.
    for subj_idx in range(n_subjects):
        # Get the train data of `subj_i`.
        subj_i = params.train.subjs[subj_idx]
        # Update optimizer learning rate if needed.
        for param_group in optimizer.param_groups: param_group["lr"] = params.train.lr_i
        # Load train dataset.
        dataset_train = utils.data.seeg.he2023xuanwu(
            base=params.train.base, subjs=[subj_i], dataset="train", n_subjects=n_subjects
        ); print("INFO: Load train set with {} samples.".format(len(dataset_train)))
        loader_train = torch.utils.data.DataLoader(
            dataset=dataset_train, batch_size=params.train.batch_size,
            shuffle=True, num_workers=0, drop_last=False, pin_memory=True
        )
        # Loop through training data
        for batch_idx, batch_data in enumerate(loader_train):
            # Get the batch data.
            X_train, y_train = batch_data
            X_train = X_train.to(device=params.model.device, dtype=torch.float32)
            y_train = y_train.to(device=params.model.device, dtype=torch.long)

            # Prepare subject ID
            batch_size = X_train.shape[0]
            subj_id = torch.zeros((batch_size, n_subjects), dtype=torch.float32, device=params.model.device)
            subj_id[:, subj_idx] = 1.

            # Forward pass
            inputs = [X_train, y_train, subj_id]
            y_pred, loss = model(inputs)

            # Backward pass
            optimizer.zero_grad()
            loss.total.backward()
            optimizer.step()

            # Calculate accuracy
            _, predicted = torch.max(y_pred.data, 1)
            train_total += y_train.size(0)
            train_correct += (predicted == y_train).sum().item()
            train_loss += loss.total.item() * y_train.size(0)

            # Log training progress
            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(loader_train):
                batch_acc = 100. * (predicted == y_train).sum().item() / y_train.size(0)
                msg = (
                    "INFO: Epoch [{}/{}], Subject [{}], Batch [{}/{}], "
                    "Loss: {:.4f}, Accuracy: {:.2f}%"
                ).format(
                    params.train.epoch + 1, params.train.n_epochs,
                    subj_i, batch_idx + 1, len(loader_train),
                    loss.total.item(), batch_acc
                )
                print(msg)

    # Calculate epoch metrics
    train_loss /= train_total
    train_acc = 100. * train_correct / train_total

    # Log epoch summary
    time_end = time.time()
    msg = (
        "INFO: Training epoch ({}) completed in {:.2f} seconds. "
        "Train Loss: {:.4f}, Train Accuracy: {:.2f}%"
    ).format(params.train.epoch, time_end - time_start, train_loss, train_acc)
    print(msg); paths.run.logger.summaries.info(msg)

# def _valid_epoch func
def _valid_epoch():
    """
    Validate model for one epoch.

    Args:
        None

    Returns:
        None
    """
    global model
    # Initialize `model.eval`.
    model.eval()
    # Initialize the record of time.
    time_start = time.time()
    # Get the number of subjects used for validation.
    n_subjects = len(params.train.subjs)
    # Initialize validation metrics
    valid_loss = 0.; valid_correct = 0; valid_total = 0
    # Execute validation loop.
    with torch.no_grad():
        for subj_idx in range(n_subjects):
            # Get the validation data of `subj_i`.
            subj_i = params.train.subjs[subj_idx]
            # Load validation dataset.
            dataset_valid = utils.data.seeg.he2023xuanwu(
                base=params.train.base, subjs=[subj_i], dataset="valid", n_subjects=n_subjects
            ); print("INFO: Load valid set with {} samples.".format(len(dataset_valid)))
            loader_valid = torch.utils.data.DataLoader(
                dataset=dataset_valid, batch_size=params.train.batch_size,
                shuffle=False, num_workers=0, drop_last=False, pin_memory=True
            )
            # Loop through validation data
            for batch_idx, batch_data in enumerate(loader_valid):
                # Get the batch data.
                X_valid, y_valid = batch_data
                X_valid = X_valid.to(device=params.model.device, dtype=torch.float32)
                y_valid = y_valid.to(device=params.model.device, dtype=torch.long)

                # Prepare subject ID
                batch_size = X_valid.shape[0]
                subj_id = torch.zeros((batch_size, n_subjects), dtype=torch.float32, device=params.model.device)
                subj_id[:, subj_idx] = 1.

                # Forward pass
                inputs = [X_valid, y_valid, subj_id]
                y_pred, loss = model(inputs)

                # Calculate metrics
                _, predicted = torch.max(y_pred.data, 1)
                valid_total += y_valid.size(0)
                valid_correct += (predicted == y_valid).sum().item()
                valid_loss += loss.total.item() * y_valid.size(0)

    # Calculate validation metrics
    valid_loss /= valid_total
    valid_acc = 100. * valid_correct / valid_total

    # Log validation summary
    time_end = time.time()
    msg = (
        "INFO: Validation epoch ({}) completed in {:.2f} seconds. "
        "Valid Loss: {:.4f}, Valid Accuracy: {:.2f}%"
    ).format(params.train.epoch, time_end - time_start, valid_loss, valid_acc)
    print(msg); paths.run.logger.summaries.info(msg)

# def _log_epoch func
def _log_epoch():
    """
    Log information of one epoch, including saving checkpoints.

    Args:
        None

    Returns:
        None
    """
    # Save checkpoint every 50 epochs and at the last epoch
    if (params.train.epoch + 1) % 50 == 0 or (params.train.epoch + 1) == params.train.n_epochs:
        ckpt_path = os.path.join(paths.run.ckpt, f"checkpoint-{params.train.epoch}.pth")
        # Handle DataParallel model
        model_to_save = model.module if hasattr(model, 'module') else model
        torch.save(model_to_save.state_dict(), ckpt_path)
        msg = f"INFO: Saved checkpoint to {ckpt_path}"
        print(msg); paths.run.logger.summaries.info(msg)

"""
arg funcs
"""
# def get_args_parser func
def get_args_parser():
    """
    Parse arguments from command line.

    Args:
        None

    Returns:
        parser: ArgumentParser - The parser of arguments.
    """
    # Initialize parser.
    parser = argparse.ArgumentParser("Fusion Classifier Training for Du-IN", add_help=False)
    # Add arguments.
    parser.add_argument("--base", type=str, default=os.getcwd())
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--subjs", type=str, default="001")
    parser.add_argument("--subj_idxs", type=str, default="0")
    parser.add_argument("--pt_multitask_ckpt", type=str, required=True,
                        help="Path to pretrained multi-task checkpoint")
    parser.add_argument("--n_epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr_min", type=float, default=1e-5)
    parser.add_argument("--lr_max", type=float, default=1e-4)
    # Freezing parameters
    parser.add_argument("--freeze_encoder", action="store_true", default=True,
                        help="Freeze encoder (SubjectBlock+Tokenizer+Encoder)")
    parser.add_argument("--no_freeze_encoder", action="store_false", dest="freeze_encoder",
                        help="Do NOT freeze encoder (end-to-end fine-tuning)")
    parser.add_argument("--freeze_task_heads", action="store_true", default=False,
                        help="Freeze task heads (semantic+visual+acoustic)")
    # Fusion head architecture
    parser.add_argument("--fusion_d_hidden", type=str, default="512,256",
                        help="Fusion head hidden dimensions (comma-separated)")
    parser.add_argument("--fusion_dropout", type=float, default=0.3,
                        help="Fusion head dropout rate")
    # Loss scale
    parser.add_argument("--cls_loss_scale", type=float, default=1.0)
    # Run script path
    parser.add_argument("--run_script", type=str, default=None)
    # Return parser.
    return parser

if __name__ == "__main__":
    import warnings
    warnings.filterwarnings("ignore")
    # Parse arguments.
    parser = get_args_parser(); args = parser.parse_args()
    # Process arguments to get list of configurations.
    seeds = [int(seed_i) for seed_i in args.seeds.split(",")]
    subjs = [subj_i for subj_i in args.subjs.split(",")]
    subj_idxs = [int(subj_idx_i) for subj_idx_i in args.subj_idxs.split(",")]

    # Determine the project root directory (go up two levels from train/duin)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

    # Get params from specified configuration.
    from params.duin_params import duin_fusion_cls_params
    # Execute experiments.
    for seed_i in seeds:
        for subj_i, subj_idx_i in zip(subjs, subj_idxs):
            # Initialize random seed.
            utils.model.torch.set_seeds(seed_i)
            # Initialize params.
            params_i = duin_fusion_cls_params(dataset="seeg_he2023xuanwu")
            # Use project root if args.base is the default, otherwise use args.base
            params_i.train.base = project_root if args.base == os.getcwd() else args.base
            params_i.train.seed = seed_i
            params_i.train.subjs = [subj_i]
            params_i.train.subj_idxs = [subj_idx_i]
            params_i.train.pt_multitask_ckpt = args.pt_multitask_ckpt
            params_i.train.n_epochs = args.n_epochs
            params_i.train.warmup_epochs = args.warmup_epochs
            params_i.train.batch_size = args.batch_size
            params_i.train.lr_factors = (args.lr_min, args.lr_max)
            params_i.train.epoch_start = 0
            params_i.train.run_script = args.run_script
            # Freezing parameters
            params_i.model.freeze_encoder = args.freeze_encoder
            params_i.model.freeze_task_heads = args.freeze_task_heads
            # Fusion head architecture
            params_i.model.fusion.d_hidden = [int(d) for d in args.fusion_d_hidden.split(",")]
            params_i.model.fusion.dropout = args.fusion_dropout
            # Loss scale
            params_i.model.cls_loss_scale = args.cls_loss_scale
            # Initialize training process.
            init(params_=params_i)
            # Execute training process.
            train()
