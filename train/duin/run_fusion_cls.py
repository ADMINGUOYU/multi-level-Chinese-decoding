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
# Data variables (loaded before init)
dataset_train = None; dataset_validation = None; dataset_test = None

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
        # We will use `_test_epoch` to test the performance of one epoch.
        _test_epoch()
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
    global model, optimizer, dataset_train
    # Initialize `model.train`.
    model.train()
    # Initialize the record of time.
    time_start = time.time()
    # Get subject info
    subj_i = params.train.subjs[0]

    # Update optimizer learning rate
    for param_group in optimizer.param_groups: param_group["lr"] = params.train.lr_i

    # Initialize training metrics
    train_loss = 0.; train_correct = 0; train_total = 0

    # Loop through training data (dataset_train is already a DataLoader)
    for batch_idx, batch_data in enumerate(dataset_train):
        # Get the batch data from FinetuneDataset
        X_train = batch_data["X"].to(device=params.model.device, dtype=torch.float32)
        y_train = batch_data["y"].to(device=params.model.device, dtype=torch.long)
        subj_id = batch_data["subj_id"].to(device=params.model.device, dtype=torch.float32)

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

        # Batch-level logging disabled (only show epoch-level results)

    # Calculate epoch metrics
    train_loss /= train_total
    train_acc = 100. * train_correct / train_total

    # Store train metrics in params for logging
    params.train.train_loss = train_loss
    params.train.train_acc = train_acc
    params.train.train_time = time.time() - time_start

# def _valid_epoch func
def _valid_epoch():
    """
    Validate model for one epoch.

    Args:
        None

    Returns:
        None
    """
    global model, dataset_validation
    # Initialize `model.eval`.
    model.eval()
    # Initialize the record of time.
    time_start = time.time()
    # Initialize validation metrics
    valid_loss = 0.; valid_correct = 0; valid_total = 0
    # Execute validation loop.
    with torch.no_grad():
        # Loop through validation data (dataset_validation is already a DataLoader)
        for batch_data in dataset_validation:
            # Get the batch data from FinetuneDataset
            X_valid = batch_data["X"].to(device=params.model.device, dtype=torch.float32)
            y_valid = batch_data["y"].to(device=params.model.device, dtype=torch.long)
            subj_id = batch_data["subj_id"].to(device=params.model.device, dtype=torch.float32)

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

    # Store validation metrics in params for logging
    params.train.valid_loss = valid_loss
    params.train.valid_acc = valid_acc

# def _test_epoch func
def _test_epoch():
    """
    Test model for one epoch.

    Args:
        None

    Returns:
        None
    """
    global model, dataset_test
    # Initialize `model.eval`.
    model.eval()
    # Initialize validation metrics
    test_loss = 0.; test_correct = 0; test_total = 0
    # Execute test loop.
    with torch.no_grad():
        # Loop through test data (dataset_test is already a DataLoader)
        for batch_data in dataset_test:
            # Get the batch data from FinetuneDataset
            X_test = batch_data["X"].to(device=params.model.device, dtype=torch.float32)
            y_test = batch_data["y"].to(device=params.model.device, dtype=torch.long)
            subj_id = batch_data["subj_id"].to(device=params.model.device, dtype=torch.float32)

            # Forward pass
            inputs = [X_test, y_test, subj_id]
            y_pred, loss = model(inputs)

            # Calculate metrics
            _, predicted = torch.max(y_pred.data, 1)
            test_total += y_test.size(0)
            test_correct += (predicted == y_test).sum().item()
            test_loss += loss.total.item() * y_test.size(0)

    # Calculate test metrics
    test_loss /= test_total
    test_acc = 100. * test_correct / test_total

    # Store test metrics in params for logging
    params.train.test_loss = test_loss
    params.train.test_acc = test_acc

# def _log_epoch func
def _log_epoch():
    """
    Log information of one epoch, including saving checkpoints.

    Args:
        None

    Returns:
        None
    """
    # Log epoch-level results (train/valid/test)
    msg = (
        f"Epoch [{params.train.epoch + 1}/{params.train.n_epochs}] ({params.train.train_time:.1f}s) | "
        f"Train Loss: {params.train.train_loss:.4f}, Acc: {params.train.train_acc:.2f}% | "
        f"Valid Loss: {params.train.valid_loss:.4f}, Acc: {params.train.valid_acc:.2f}% | "
        f"Test Loss: {params.train.test_loss:.4f}, Acc: {params.train.test_acc:.2f}%"
    )
    print(msg); paths.run.logger.summaries.info(msg)

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

"""
dataset funcs
"""
class FinetuneDataset(torch.utils.data.Dataset):
    """
    Brain signal finetune dataset for fusion classification.
    Similar to run_align_vis.py but adapted for integer label classification.
    """

    def __init__(self, data_items, use_aug=False, **kwargs):
        """
        Initialize `FinetuneDataset` object.

        Args:
            data_items: list - The list of data items, including [X,y,subj_id].
            use_aug: bool - The flag that indicates whether enable augmentations.
            kwargs: dict - The arguments related to initialize `torch.utils.data.Dataset`-style object.

        Returns:
            None
        """
        super(FinetuneDataset, self).__init__(**kwargs)
        self.data_items = data_items
        self.use_aug = use_aug
        # Initialize variables.
        self._init_dataset()

    def _init_dataset(self):
        """Initialize the configuration of dataset."""
        # Initialize the maximum shift steps for augmentation.
        self.max_steps = self.data_items[0].X.shape[1] // 10

    def __len__(self):
        """Get the number of samples of dataset."""
        return len(self.data_items)

    def __getitem__(self, index):
        """
        Get the data item corresponding to data index.

        Args:
            index: int - The index of data item to get.

        Returns:
            data: dict - The data item dictionary.
        """
        # Load data item
        data_item = self.data_items[index]
        # X - (n_channels, seq_len); y - (scalar integer); subj_id - (n_subjects,)
        X = data_item.X
        y = data_item.y
        subj_id = data_item.subj_id

        # Execute data augmentations
        if self.use_aug:
            # Randomly shift `X` according to `max_steps`.
            X_shifted = np.zeros(X.shape, dtype=X.dtype)
            n_steps = np.random.choice((np.arange(2 * self.max_steps + 1, dtype=np.int64) - self.max_steps))
            if n_steps > 0:
                X_shifted[:,n_steps:] = X[:,:-n_steps]
            elif n_steps < 0:
                X_shifted[:,:n_steps] = X[:,-n_steps:]
            else:
                pass
            X = X_shifted

        # Construct the data dict
        data = {
            "X": torch.from_numpy(X.T).to(dtype=torch.float32),  # Transpose to (seq_len, n_channels)
            "y": torch.tensor(y, dtype=torch.long),  # Integer label for classification
            "subj_id": torch.from_numpy(subj_id).to(dtype=torch.float32),
        }
        return data

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

            # ===== LOAD PARAMS FROM PRETRAINED MULTITASK CHECKPOINT =====
            # Extract the params path from the checkpoint path
            # e.g., /path/to/summaries/2025-11-17/1/train/ckpt/checkpoint-299.pth
            #   -> /path/to/summaries/2025-11-17/1/save/params
            pt_ckpt_path = os.path.abspath(args.pt_multitask_ckpt)
            pt_params_path = os.path.normpath(os.path.join(os.path.dirname(pt_ckpt_path), "..", "..", "save", "params"))

            print(f"[DEBUG] Checkpoint path: {pt_ckpt_path}")
            print(f"[DEBUG] Looking for params at: {pt_params_path}")
            print(f"[DEBUG] Params file exists: {os.path.exists(pt_params_path)}")

            if not os.path.exists(pt_params_path):
                raise FileNotFoundError(
                    f"ERROR: Pretrained multitask params not found at: {pt_params_path}\n"
                    f"The fusion classifier requires the params file from the pretrained multitask model.\n"
                    f"Please ensure the checkpoint was trained using run_multitask.py which saves params to save/params."
                )

            # Load pretrained multitask params
            params_pt = load_pickle(pt_params_path)
            print(f"[INFO] ✓ Loaded pretrained multitask params from: {pt_params_path}")

            # ===== EXTRACT ACTUAL ARCHITECTURE FROM CHECKPOINT WEIGHTS =====
            # The saved params file may have incorrect values due to a bug in run_multitask.py
            # So we extract the true architecture directly from the checkpoint weight dimensions
            print(f"[INFO] Extracting actual architecture from checkpoint weights...")
            checkpoint = torch.load(pt_ckpt_path, map_location='cpu')

            # Extract subject block dimensions from subj_layer weights
            # W.weight shape: [d_input * d_output, n_subjects]
            # B.weight shape: [d_output, n_subjects] (if use_bias=True)
            if 'subj_block.subj_layer.W.weight' in checkpoint:
                n_subjects_actual = checkpoint['subj_block.subj_layer.W.weight'].shape[1]
                d_input_times_d_output = checkpoint['subj_block.subj_layer.W.weight'].shape[0]
            else:
                raise ValueError("Cannot find 'subj_block.subj_layer.W.weight' in checkpoint")

            # Extract d_output from B.weight (more reliable than calculating from W.weight)
            if 'subj_block.subj_layer.B.weight' in checkpoint:
                d_output_actual = checkpoint['subj_block.subj_layer.B.weight'].shape[0]
                # Calculate actual n_channels (d_input) from W.weight and B.weight
                n_channels_actual = d_input_times_d_output // d_output_actual
            else:
                # If no B.weight (use_bias=False), use params file value
                n_channels_actual = params_pt.model.n_channels
                d_output_actual = d_input_times_d_output // n_channels_actual

            # Extract token_len and d_model from emb_time.time_encodings: shape [token_len, d_model]
            if 'emb_time.time_encodings' in checkpoint:
                token_len_actual = checkpoint['emb_time.time_encodings'].shape[0]
                d_model_actual = checkpoint['emb_time.time_encodings'].shape[1]
            else:
                raise ValueError("Cannot find 'emb_time.time_encodings' in checkpoint")

            # Extract semantic_align hidden dims from weights
            if 'semantic_head.align_head.1.0.weight' in checkpoint:
                semantic_hidden_1 = checkpoint['semantic_head.align_head.1.0.weight'].shape[0]
                semantic_input_dim = checkpoint['semantic_head.align_head.1.0.weight'].shape[1]
            if 'semantic_head.align_head.2.0.weight' in checkpoint:
                semantic_hidden_2 = checkpoint['semantic_head.align_head.2.0.weight'].shape[0]

            # Extract visual_align hidden dims from weights
            if 'visual_head.align_head.1.0.weight' in checkpoint:
                visual_hidden_1 = checkpoint['visual_head.align_head.1.0.weight'].shape[0]
            if 'visual_head.align_head.2.0.weight' in checkpoint:
                visual_hidden_2 = checkpoint['visual_head.align_head.2.0.weight'].shape[0]

            print(f"[INFO] Actual architecture from checkpoint:")
            print(f"  - n_subjects: {n_subjects_actual} (params file had: {params_pt.model.n_subjects})")
            print(f"  - n_channels: {n_channels_actual}")
            print(f"  - d_output (SubjectBlock): {d_output_actual} (params file had: {params_pt.model.subj.d_output})")
            print(f"  - d_model: {d_model_actual} (params file had: {params_pt.model.encoder.d_model})")
            print(f"  - token_len: {token_len_actual} (params file had: {params_pt.model.tokenizer.token_len})")
            print(f"  - semantic_align.d_hidden: [{semantic_hidden_1}, {semantic_hidden_2}]")
            print(f"  - visual_align.d_hidden: [{visual_hidden_1}, {visual_hidden_2}]")

            # Override params_pt with actual values from checkpoint
            params_pt.model.n_subjects = n_subjects_actual
            params_pt.model.subj.n_subjects = n_subjects_actual
            params_pt.model.subj.d_input = n_channels_actual
            params_pt.model.subj.d_output = d_output_actual
            params_pt.model.encoder.d_model = d_model_actual
            params_pt.model.encoder.emb_len = token_len_actual
            params_pt.model.tokenizer.token_len = token_len_actual
            params_pt.model.tokenizer.d_model = d_model_actual
            params_pt.model.tokenizer.d_neural = d_output_actual

            # Update alignment head parameters
            # d_feature = token_len * d_model (flattened embedding)
            align_d_feature = token_len_actual * d_model_actual
            params_pt.model.semantic_align.d_feature = align_d_feature
            params_pt.model.semantic_align.d_hidden = [semantic_hidden_1, semantic_hidden_2]

            params_pt.model.visual_align.d_feature = align_d_feature
            params_pt.model.visual_align.d_hidden = [visual_hidden_1, visual_hidden_2]

            # Update acoustic classification head parameters
            params_pt.model.acoustic_cls.d_model = d_model_actual
            params_pt.model.acoustic_cls.emb_len = token_len_actual

            # Update VQ and contrastive parameters
            params_pt.model.vq.d_model = d_model_actual
            params_pt.model.contra.d_model = d_model_actual

            # Initialize fusion classifier params based on pretrained params
            params_i = duin_fusion_cls_params(dataset="seeg_he2023xuanwu")

            # Copy all multitask-related parameters from pretrained params
            # These MUST match the checkpoint exactly
            params_i.model.n_subjects = params_pt.model.n_subjects
            params_i.model.n_channels = params_pt.model.n_channels
            # Calculate actual seq_len from checkpoint's token_len (not from params file which may be wrong)
            params_i.model.seq_len = token_len_actual * params_pt.model.seg_len
            params_i.model.seg_len = params_pt.model.seg_len

            # Copy SubjectBlock params
            params_i.model.subj = cp.deepcopy(params_pt.model.subj)

            # Copy Tokenizer params
            params_i.model.tokenizer = cp.deepcopy(params_pt.model.tokenizer)

            # Copy Encoder params
            params_i.model.encoder = cp.deepcopy(params_pt.model.encoder)

            # Copy VQ params
            params_i.model.vq = cp.deepcopy(params_pt.model.vq)

            # Copy Contrastive params
            params_i.model.contra = cp.deepcopy(params_pt.model.contra)

            # Copy task-specific head params
            params_i.model.semantic_align = cp.deepcopy(params_pt.model.semantic_align)
            params_i.model.visual_align = cp.deepcopy(params_pt.model.visual_align)
            params_i.model.acoustic_cls = cp.deepcopy(params_pt.model.acoustic_cls)

            # Copy multitask-specific parameters
            params_i.model.task_weight_semantic = params_pt.model.task_weight_semantic
            params_i.model.task_weight_visual = params_pt.model.task_weight_visual
            params_i.model.task_weight_acoustic = params_pt.model.task_weight_acoustic
            params_i.model.use_uncertainty_weighting = params_pt.model.use_uncertainty_weighting
            params_i.model.acoustic_use_contra = params_pt.model.acoustic_use_contra

            # Debug: print key dimensions to verify they match
            print(f"[INFO] ✓ Loaded architecture from pretrained multitask model:")
            print(f"  - n_subjects: {params_i.model.n_subjects}")
            print(f"  - n_channels: {params_i.model.n_channels}")
            print(f"  - seq_len: {params_i.model.seq_len} (calculated from token_len={token_len_actual} × seg_len={params_i.model.seg_len})")
            print(f"  - subj.d_input: {params_i.model.subj.d_input}")
            print(f"  - subj.d_output: {params_i.model.subj.d_output}")
            print(f"  - tokenizer.d_neural: {params_i.model.tokenizer.d_neural}")
            print(f"  - tokenizer.d_model: {params_i.model.tokenizer.d_model}")
            print(f"  - tokenizer.token_len: {params_i.model.tokenizer.token_len}")
            print(f"  - encoder.d_model: {params_i.model.encoder.d_model}")
            print(f"  - encoder.emb_len: {params_i.model.encoder.emb_len}")
            print(f"  - semantic_align.d_feature: {params_i.model.semantic_align.d_feature}")
            print(f"  - semantic_align.d_hidden: {params_i.model.semantic_align.d_hidden}")
            print(f"  - visual_align.d_feature: {params_i.model.visual_align.d_feature}")
            print(f"  - visual_align.d_hidden: {params_i.model.visual_align.d_hidden}")
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

            # ===== SET DEVICE =====
            # Initialize model device (use GPU if available)
            if torch.cuda.is_available():
                params_i.model.device = torch.device("cuda:{:d}".format(0))
                print(f"[INFO] Using device: {params_i.model.device}")
                print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
                print(f"[INFO] GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
            else:
                params_i.model.device = torch.device("cpu")
                print(f"[WARNING] CUDA not available, using CPU")

            # ===== LOAD DATA (BEFORE INIT) =====
            # Define subject configurations (same as run_align_vis.py)
            subjs_cfg = utils.DotDict({
                "001": utils.DotDict({
                    "name": "001", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "001"),
                    "ch_names": ["SM8", "SM9", "SM7", "SM11", "P4", "SM10", "SM6", "P3", "SM5", "CI9"],
                }),
                "002": utils.DotDict({
                    "name": "002", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "002"),
                    "ch_names": ["TI'2", "TI'3", "TI'1", "TI'6", "TI'4", "TI'7", "ST'3", "ST'2", "ST'4", "FP'4"],
                }),
                "003": utils.DotDict({
                    "name": "003", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "003"),
                    "ch_names": ["ST3", "ST1", "ST2", "ST9", "TI'4", "TI'3", "ST4", "TI'2", "ST7", "TI'8"] ,
                }),
                "004": utils.DotDict({
                    "name": "004", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "004"),
                    "ch_names": ["D12", "D13", "C4", "C3", "D11", "D14", "D10", "D9", "D5", "C15"],
                }),
                "005": utils.DotDict({
                    "name": "005", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "005"),
                    "ch_names": ["E8", "E9", "E6", "E7", "E11", "E12", "E5", "E10", "C10", "E4"],
                }),
                "006": utils.DotDict({
                    "name": "006", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "006"),
                    "ch_names": ["D3", "D1", "D6", "D2", "D5", "D4", "D7", "D8", "G8", "E13"],
                }),
                "007": utils.DotDict({
                    "name": "007", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "007"),
                    "ch_names": ["H2", "H4", "H3", "H1", "H6", "H5", "E4", "H7", "C13", "E5"],
                }),
                "008": utils.DotDict({
                    "name": "008", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "008"),
                    "ch_names": ["TI3", "TI4", "TI2", "TI5", "B9", "TI6", "TI7", "TI9", "TI10", "B5"],
                }),
                "009": utils.DotDict({
                    "name": "009", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "009"),
                    "ch_names": ["K9", "K8", "K6", "K7", "K11", "K10", "K5", "K4", "K3", "I9"],
                }),
                "010": utils.DotDict({
                    "name": "010", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "010"),
                    "ch_names": ["PI5", "PI6", "PI7", "PI8", "PI1", "PI9", "PI2", "SM2", "SP3", "PI4"],
                }),
                "011": utils.DotDict({
                    "name": "011", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "011"),
                    "ch_names": ["T2", "T3", "C9", "T4", "T5", "C7", "C8", "T1", "s1", "C4"],
                }),
                "012": utils.DotDict({
                    "name": "012", "path": os.path.join(params_i.train.base, "data", "seeg.he2023xuanwu", "012"),
                    "ch_names": ["TI'4", "TI'2", "TI'3", "TI'5", "TI'8", "TI'6", "TI'7", "TO'9", "P'5", "TO'8"],
                }),
            })

            # Get configuration for current subject
            subj_cfg_i = subjs_cfg[subj_i]

            # Create load_params manually (like run_cls.py and run_align_vis.py)
            load_params = utils.DotDict({
                "type": "bipolar_default",
                "task": "word_recitation",
                "use_align": False,
                "resample_rate": 1000,
            })

            # Load data from specified subject
            func = getattr(getattr(utils.data.seeg.he2023xuanwu, load_params.task), "load_subj_{}".format(load_params.type))
            dataset = func(subj_cfg_i.path, ch_names=subj_cfg_i.ch_names, use_align=load_params.use_align)
            X = dataset.X_s.astype(np.float32); y = dataset.y.astype(np.int64)

            # Preprocess data
            if load_params.type.startswith("bipolar"):
                sample_rate = 1000
                X = sp.signal.resample(X, int(np.round(X.shape[1] / (sample_rate / load_params.resample_rate))), axis=1)

                # CRITICAL: Extract time window to match checkpoint's seq_len
                # Checkpoint has token_len=25, seg_len=100 → seq_len must be 2500
                # Calculate required seq_len from checkpoint architecture
                required_seq_len = token_len_actual * 100  # token_len * seg_len
                time_window_end = (-0.5) + (required_seq_len / load_params.resample_rate)  # -0.5 + 2.5 = 2.0

                X = X[:,int(np.round((-0.5 - (-0.5)) * load_params.resample_rate)):
                       int(np.round((time_window_end - (-0.5)) * load_params.resample_rate)),:]

                print(f"[INFO] Data window: -0.5s to {time_window_end}s, seq_len: {X.shape[1]}, expected: {required_seq_len}")
                assert X.shape[1] == required_seq_len, f"Data seq_len {X.shape[1]} doesn't match checkpoint {required_seq_len}"

                X = (X - np.mean(X, axis=(0,1), keepdims=True)) / np.std(X, axis=(0,1), keepdims=True)
            else:
                raise ValueError("ERROR: Unknown type {} of dataset.".format(load_params.type))

            # Train/test split
            train_ratio = params_i.train.train_ratio
            train_idxs = []; test_idxs = []
            for label_i in sorted(set(y)):
                label_idxs = np.where(y == label_i)[0].tolist()
                train_idxs.extend(label_idxs[:int(train_ratio * len(label_idxs))])
                test_idxs.extend(label_idxs[int(train_ratio * len(label_idxs)):])
            train_idxs = np.array(train_idxs, dtype=np.int64); test_idxs = np.array(test_idxs, dtype=np.int64)
            X_train = X[train_idxs,:,:]; y_train = y[train_idxs]
            X_test = X[test_idxs,:,:]; y_test = y[test_idxs]

            # Transform y to sorted order (class indices 0-60)
            labels = sorted(set(y_train))
            y_train_idx = np.array([labels.index(y_i) for y_i in y_train], dtype=np.int64)
            y_test_idx = np.array([labels.index(y_i) for y_i in y_test], dtype=np.int64)

            # Further split test-set into validation-set & test-set
            validation_idxs = np.random.choice(np.arange(X_test.shape[0]), size=int(X_test.shape[0]/2), replace=False)
            validation_mask = np.zeros((X_test.shape[0],), dtype=np.bool_); validation_mask[validation_idxs] = True
            X_validation = X_test[validation_mask,:,:]; y_validation = y_test_idx[validation_mask]
            X_test = X_test[~validation_mask,:,:]; y_test = y_test_idx[~validation_mask]

            # Construct subj_id arrays
            n_subjects = params_i.model.n_subjects
            subj_id_train = np.array([np.eye(n_subjects)[subj_idx_i] for _ in range(X_train.shape[0])])
            subj_id_validation = np.array([np.eye(n_subjects)[subj_idx_i] for _ in range(X_validation.shape[0])])
            subj_id_test = np.array([np.eye(n_subjects)[subj_idx_i] for _ in range(X_test.shape[0])])

            # Log data preparation
            msg = (
                "INFO: Data preparation for subject ({}) complete, with train-set ({}) & validation-set ({}) & test-set ({})."
            ).format(subj_i, X_train.shape, X_validation.shape, X_test.shape)
            print(msg)

            # Construct FinetuneDataset
            dataset_train = FinetuneDataset(data_items=[utils.DotDict({
                "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
            }) for X_i, y_i, subj_id_i in zip(X_train, y_train_idx, subj_id_train)], use_aug=True)

            dataset_validation = FinetuneDataset(data_items=[utils.DotDict({
                "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
            }) for X_i, y_i, subj_id_i in zip(X_validation, y_validation, subj_id_validation)], use_aug=False)

            dataset_test = FinetuneDataset(data_items=[utils.DotDict({
                "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
            }) for X_i, y_i, subj_id_i in zip(X_test, y_test, subj_id_test)], use_aug=False)

            # Create DataLoaders
            dataset_train = torch.utils.data.DataLoader(dataset_train,
                batch_size=params_i.train.batch_size, shuffle=True, drop_last=False)
            dataset_validation = torch.utils.data.DataLoader(dataset_validation,
                batch_size=params_i.train.batch_size, shuffle=False, drop_last=False)
            dataset_test = torch.utils.data.DataLoader(dataset_test,
                batch_size=params_i.train.batch_size, shuffle=False, drop_last=False)

            print(f"INFO: DataLoaders created - Train: {len(dataset_train)} batches, Valid: {len(dataset_validation)} batches, Test: {len(dataset_test)} batches")

            # Initialize training process.
            init(params_=params_i)
            # Execute training process.
            train()
