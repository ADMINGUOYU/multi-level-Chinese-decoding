#!/usr/bin/env python3
"""
Created on acoustic tone classification

@author: Claude Code (adapted from run_cls.py)
"""
import torch
import os, time, sys
import argparse
import copy as cp
import numpy as np
import scipy as sp
import unicodedata
from collections import Counter
from torchinfo import summary
from torch.utils.tensorboard import SummaryWriter
# local dep
if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.join(os.pardir, os.pardir))
import utils; import utils.model.torch; import utils.data.seeg
from utils.data import load_pickle
from models.duin import duin_acoustic_cls as duin_model
from GT_embeddings.load_GT import load_npz_acoustic

__all__ = [
    "init",
    "train",
]

# GPU DEBUGGING: Disable cuDNN to test if it causes GPU training failure
import torch.backends.cudnn as cudnn
cudnn.enabled = False
cudnn.benchmark = False
cudnn.deterministic = True

# Global variables.
params = None; paths = None
model = None; optimizer = None

"""
init funcs
"""
# def init func
def init(params_):
    """
    Initialize `duin_acoustic_cls` training variables.

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

# def _init_model func
def _init_model():
    """
    Initialize model used in the training process.

    Args:
        None

    Returns:
        None
    """
    global params
    ## Initialize torch configuration.
    # Not set random seed, should be done before initializing `model`.
    torch.set_default_dtype(getattr(torch, params._precision))
    # Set the internal precision of float32 matrix multiplications.
    torch.set_float32_matmul_precision("high")

# def _init_train func
def _init_train():
    """
    Initialize the training process.

    Args:
        None

    Returns:
        None
    """
    pass

"""
data funcs
"""
# def load_data func
def load_data(load_params):
    """
    Load data from specified dataset.

    Args:
        load_params: DotDict - The load parameters of specified dataset.

    Returns:
        dataset_train: torch.utils.data.DataLoader - The train dataset, including (X_train, y_train).
        dataset_validation: torch.utils.data.DataLoader - The validation dataset, including (X_validation, y_validation).
        dataset_test: torch.utils.data.DataLoader - The test dataset, including (X_test, y_test).
    """
    global params
    # Load data from specified dataset.
    try:
        func = getattr(sys.modules[__name__], "_".join(["_load_data", params.train.dataset]))
        dataset_train, dataset_validation, dataset_test = func(load_params)
    except Exception:
        raise ValueError((
            "ERROR: Unknown dataset type {} in train.duin.run_acoustic_cls."
        ).format(params.train.dataset))
    # Return the final `dataset_train` & `dataset_validation` & `dataset_test`.
    return dataset_train, dataset_validation, dataset_test

# def _load_data_seeg_he2023xuanwu func
def _load_data_seeg_he2023xuanwu(load_params):
    """
    Load seeg data from the specified subject in `seeg_he2023xuanwu`.

    Args:
        load_params: DotDict - The load parameters of specified dataset.

    Returns:
        dataset_train: torch.utils.data.DataLoader - The train dataset, including (X_train, y_train).
        dataset_validation: torch.utils.data.DataLoader - The validation dataset, including (X_validation, y_validation).
        dataset_test: torch.utils.data.DataLoader - The test dataset, including (X_test, y_test).
    """
    global params, paths

    # --- Load acoustic tone labels ---
    acoustic_path = os.path.join(params.train.base, "GT_embeddings", "Duin_Acoustic_label.npz")
    acoustic_chars, acoustic_labels = load_npz_acoustic(acoustic_path)
    print(f"[INFO] Loaded acoustic labels: shape={acoustic_labels.shape}, chars={len(acoustic_chars)}")
    print(f"[INFO] Sample labels: {acoustic_labels[:5]}")

    # Normalize text helper function
    def _norm_text(s: str) -> str:
        return unicodedata.normalize("NFKC", str(s)).strip()

    # Get label order from dataset (same as semantic/visual align approach)
    subjs_cfg = load_params.subjs_cfg
    label_name_set = set()
    subj0_path = subjs_cfg[0].path
    task_dir = os.path.join(subj0_path, "word-recitation")
    dataset_dir_name = f"dataset.bipolar.default.{'aligned' if load_params.use_align else 'unaligned'}"

    for run_i in os.listdir(task_dir):
        data_pkl = os.path.join(task_dir, run_i, dataset_dir_name, "data")
        if not os.path.exists(data_pkl):
            continue
        dataset_data_i = load_pickle(data_pkl)
        for sample in dataset_data_i:
            if hasattr(sample, "name"):
                label_name_set.add(_norm_text(sample.name))

    # Get label order (sorted by name, consistent with dataset loading)
    label_order = sorted(label_name_set)
    print(f"[INFO] Dataset label order size={len(label_order)}, sample={label_order[:10]}")

    # Align acoustic labels to dataset label order
    acoustic_chars_norm = [_norm_text(w) for w in acoustic_chars]
    word_to_idx = {w: i for i, w in enumerate(acoustic_chars_norm)}

    # Create aligned tone label table: (n_labels, 2) where 2 is [tone1, tone2]
    aligned_tone_labels = np.zeros((len(label_order), 2), dtype=np.int64)
    missing = []
    for i, name in enumerate(label_order):
        j = word_to_idx.get(_norm_text(name), None)
        if j is None:
            missing.append(name)
            # Use class 0 as placeholder for missing labels
            aligned_tone_labels[i, :] = 0
        else:
            # acoustic_labels[j] contains [tone1_class, tone2_class] where classes are 1-5
            # Convert to 0-indexed: subtract 1
            aligned_tone_labels[i, :] = acoustic_labels[j] - 1

    if missing:
        print(f"[WARNING] {len(missing)} labels not found in acoustic table. e.g. {missing[:10]}")

    print("[INFO] ---- Alignment Check ----")
    for i in range(min(10, len(label_order))):
        label = label_order[i]
        tones = aligned_tone_labels[i]
        print(f"  Dataset label[{i}]: {label:>6s}  →  Tones: {tones}")

    # Store in params
    params.model.tone_table = aligned_tone_labels
    params.model.tone_words = np.array(label_order)

    # Initialize subjs_cfg.
    subjs_cfg = load_params.subjs_cfg
    # Initialize `n_subjects` & `n_subjects` & `subj_idxs` & `seq_len` & `n_labels` from `load_params`.
    n_channels = load_params.n_channels if load_params.n_channels is not None else None
    n_subjects = load_params.n_subjects if load_params.n_subjects is not None else len(subjs_cfg)
    subj_idxs = load_params.subj_idxs if load_params.subj_idxs is not None else np.arange(n_subjects)
    seq_len = None; n_labels = None
    # Initialize `Xs_*` & `ys_*` & `subj_ids_*`, then load them.
    Xs_train = []; ys_train_tone1 = []; ys_train_tone2 = []; subj_ids_train = []
    Xs_validation = []; ys_validation_tone1 = []; ys_validation_tone2 = []; subj_ids_validation = []
    Xs_test = []; ys_test_tone1 = []; ys_test_tone2 = []; subj_ids_test = []

    for subj_idx, subj_cfg_i in zip(subj_idxs, subjs_cfg):
        # Load data from specified subject run.
        func = getattr(getattr(utils.data.seeg.he2023xuanwu, load_params.task), "load_subj_{}".format(load_params.type))
        dataset = func(subj_cfg_i.path, ch_names=subj_cfg_i.ch_names, use_align=load_params.use_align)
        X = dataset.X_s.astype(np.float32); y = dataset.y.astype(np.int64)
        # If the type of dataset is `bipolar`.
        if load_params.type.startswith("bipolar"):
            # Truncate `X` to let them have the same length.
            X = X
            # Resample the original data to the specified `resample_rate`.
            sample_rate = 1000; X = sp.signal.resample(X, int(np.round(X.shape[1] /\
                (sample_rate / load_params.resample_rate))), axis=1)
            # Truncate data according to epoch range (-0.2,1.0), the original epoch range is (-0.5,2.0).
            X = X[:,int(np.round((-0.5 - (-0.5)) * load_params.resample_rate)):\
                int(np.round((2.5 - (-0.5)) * load_params.resample_rate)),:]
            # Do Z-score for each channel.
            X = (X - np.mean(X, axis=(0,1), keepdims=True)) / np.std(X, axis=(0,1), keepdims=True)
        # Get unknown type of dataset.
        else:
            raise ValueError("ERROR: Unknown type {} of dataset.".format(load_params.type))
        # Initialize trainset & testset.
        # X - (n_samples, seq_len, n_channels); y - (n_samples,)
        train_ratio = params.train.train_ratio; train_idxs = []; test_idxs = []
        for label_i in sorted(set(y)):
            label_idxs = np.where(y == label_i)[0].tolist()
            train_idxs.extend(label_idxs[:int(train_ratio * len(label_idxs))])
            test_idxs.extend(label_idxs[int(train_ratio * len(label_idxs)):])
        for train_idx in train_idxs: assert train_idx not in test_idxs
        train_idxs = np.array(train_idxs, dtype=np.int64); test_idxs = np.array(test_idxs, dtype=np.int64)
        X_train = X[train_idxs,:,:]; y_train = y[train_idxs]; X_test = X[test_idxs,:,:]; y_test = y[test_idxs]
        # Check whether trainset & testset both have data items.
        if len(X_train) == 0 or len(X_test) == 0: return ([], []), ([], [])
        # Make sure there is no overlap between X_train & X_test.
        samples_same = None; n_samples = 10; assert X_train.shape[1] == X_test.shape[1]
        for _ in range(n_samples):
            sample_idx = np.random.randint(X_train.shape[1])
            sample_same_i = np.intersect1d(X_train[:,sample_idx,0], X_test[:,sample_idx,0], return_indices=True)[-1].tolist()
            samples_same = set(sample_same_i) if samples_same is None else set(sample_same_i) & samples_same
        assert len(samples_same) == 0
        # Check whether labels are enough, then transform y to sorted order.
        assert len(set(y_train)) == len(set(y_test)); labels = sorted(set(y_train))

        # Convert labels to acoustic tone labels (tone1 and tone2)
        y_train_idx = np.array([labels.index(y_i) for y_i in y_train], dtype=np.int64)
        y_test_idx = np.array([labels.index(y_i) for y_i in y_test], dtype=np.int64)

        # Get tone labels: tone_table is (n_labels, 2) with [tone1, tone2]
        y_train_tone1_raw = aligned_tone_labels[y_train_idx, 0]  # (n_train,) with class indices
        y_train_tone2_raw = aligned_tone_labels[y_train_idx, 1]  # (n_train,) with class indices
        y_test_tone1_raw = aligned_tone_labels[y_test_idx, 0]    # (n_test,) with class indices
        y_test_tone2_raw = aligned_tone_labels[y_test_idx, 1]    # (n_test,) with class indices

        # Convert to one-hot for model (5 classes for each tone)
        y_train_tone1 = np.eye(5)[y_train_tone1_raw]  # (n_train, 5)
        y_train_tone2 = np.eye(5)[y_train_tone2_raw]  # (n_train, 5)
        y_test_tone1 = np.eye(5)[y_test_tone1_raw]    # (n_test, 5)
        y_test_tone2 = np.eye(5)[y_test_tone2_raw]    # (n_test, 5)

        # Execute sample permutation. We only shuffle along the axis.
        if load_params.permutation:
            perm_idx = np.random.permutation(len(y_train_tone1))
            y_train_tone1 = y_train_tone1[perm_idx]
            y_train_tone2 = y_train_tone2[perm_idx]

        # Further split test-set into validation-set & test-set.
        validation_idxs = np.random.choice(np.arange(X_test.shape[0]), size=int(X_test.shape[0]/2), replace=False)
        validation_mask = np.zeros((X_test.shape[0],), dtype=np.bool_); validation_mask[validation_idxs] = True
        X_validation = X_test[validation_mask,:,:]; y_validation_tone1 = y_test_tone1[validation_mask,:]
        y_validation_tone2 = y_test_tone2[validation_mask,:]
        X_test = X_test[~validation_mask,:,:]; y_test_tone1 = y_test_tone1[~validation_mask,:]
        y_test_tone2 = y_test_tone2[~validation_mask,:]

        # Construct `subj_id_*` according to `subj_idx`.
        # subj_id - (n_samples, n_subjects)
        subj_id_train = np.array([np.eye(n_subjects)[subj_idx] for _ in range(X_train.shape[0])])
        subj_id_validation = np.array([np.eye(n_subjects)[subj_idx] for _ in range(X_validation.shape[0])])
        subj_id_test = np.array([np.eye(n_subjects)[subj_idx] for _ in range(X_test.shape[0])])
        # Log information of data loading.
        msg = (
            "INFO: Data preparation for subject ({}) complete, with train-set ({}) & validation-set ({}) & test-set ({})."
        ).format(subj_cfg_i.name, X_train.shape, X_validation.shape, X_test.shape)
        print(msg); paths.run.logger.summaries.info(msg)

        # Append `X_*` & `y_*` & `subj_id_*` to `Xs_*` & `ys_*` & `subj_ids_*`.
        Xs_train.append(X_train); ys_train_tone1.append(y_train_tone1); ys_train_tone2.append(y_train_tone2)
        subj_ids_train.append(subj_id_train)
        Xs_validation.append(X_validation); ys_validation_tone1.append(y_validation_tone1); ys_validation_tone2.append(y_validation_tone2)
        subj_ids_validation.append(subj_id_validation)
        Xs_test.append(X_test); ys_test_tone1.append(y_test_tone1); ys_test_tone2.append(y_test_tone2)
        subj_ids_test.append(subj_id_test)
        # Update `n_channels` & `seq_len` & `n_labels`.
        n_channels = max(X.shape[-1], n_channels) if n_channels is not None else X.shape[-1]
        seq_len = X.shape[-2] if seq_len is None else seq_len; assert seq_len == X.shape[-2]
        n_labels = len(labels) if n_labels is None else n_labels; assert n_labels == len(labels)

    # Check `n_channels` according to `load_params`.
    if load_params.n_channels is not None: assert n_channels == load_params.n_channels
    # Update `Xs_*` with `n_channels`.
    Xs_train = [np.concatenate([X_train_i,
        np.zeros((*X_train_i.shape[:-1], (n_channels - X_train_i.shape[-1])), dtype=X_train_i.dtype)
    ], axis=-1) for X_train_i in Xs_train]
    Xs_validation = [np.concatenate([X_validation_i,
        np.zeros((*X_validation_i.shape[:-1], (n_channels - X_validation_i.shape[-1])), dtype=X_validation_i.dtype)
    ], axis=-1) for X_validation_i in Xs_validation]
    Xs_test = [np.concatenate([X_test_i,
        np.zeros((*X_test_i.shape[:-1], (n_channels - X_test_i.shape[-1])), dtype=X_test_i.dtype)
    ], axis=-1) for X_test_i in Xs_test]
    # Combine different datasets into one dataset.
    Xs_train = np.concatenate(Xs_train, axis=0)
    ys_train_tone1 = np.concatenate(ys_train_tone1, axis=0)
    ys_train_tone2 = np.concatenate(ys_train_tone2, axis=0)
    subj_ids_train = np.concatenate(subj_ids_train, axis=0)
    Xs_validation = np.concatenate(Xs_validation, axis=0)
    ys_validation_tone1 = np.concatenate(ys_validation_tone1, axis=0)
    ys_validation_tone2 = np.concatenate(ys_validation_tone2, axis=0)
    subj_ids_validation = np.concatenate(subj_ids_validation, axis=0)
    Xs_test = np.concatenate(Xs_test, axis=0)
    ys_test_tone1 = np.concatenate(ys_test_tone1, axis=0)
    ys_test_tone2 = np.concatenate(ys_test_tone2, axis=0)
    subj_ids_test = np.concatenate(subj_ids_test, axis=0)
    # Shuffle dataset to fuse different subjects.
    train_idxs = np.arange(Xs_train.shape[0]); np.random.shuffle(train_idxs)
    validation_idxs = np.arange(Xs_validation.shape[0]); np.random.shuffle(validation_idxs)
    test_idxs = np.arange(Xs_test.shape[0]); np.random.shuffle(test_idxs)
    Xs_train = Xs_train[train_idxs,...]; ys_train_tone1 = ys_train_tone1[train_idxs,...]; ys_train_tone2 = ys_train_tone2[train_idxs,...]
    subj_ids_train = subj_ids_train[train_idxs,...]
    Xs_validation = Xs_validation[validation_idxs,...]; ys_validation_tone1 = ys_validation_tone1[validation_idxs,...]
    ys_validation_tone2 = ys_validation_tone2[validation_idxs,...]; subj_ids_validation = subj_ids_validation[validation_idxs,...]
    Xs_test = Xs_test[test_idxs,...]; ys_test_tone1 = ys_test_tone1[test_idxs,...]; ys_test_tone2 = ys_test_tone2[test_idxs,...]
    subj_ids_test = subj_ids_test[test_idxs,...]
    # Log information of data loading.
    msg = (
        "INFO: Data preparation complete, with train-set ({}) & validation-set ({}) & test-set ({})."
    ).format(Xs_train.shape, Xs_validation.shape, Xs_test.shape)
    print(msg); paths.run.logger.summaries.info(msg)
    # Construct dataset from data items.
    dataset_train = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y_tone1": y_tone1_i, "y_tone2": y_tone2_i, "subj_id": subj_id_i,
    }) for X_i, y_tone1_i, y_tone2_i, subj_id_i in zip(Xs_train, ys_train_tone1, ys_train_tone2, subj_ids_train)], use_aug=True)
    dataset_validation = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y_tone1": y_tone1_i, "y_tone2": y_tone2_i, "subj_id": subj_id_i,
    }) for X_i, y_tone1_i, y_tone2_i, subj_id_i in zip(Xs_validation, ys_validation_tone1, ys_validation_tone2, subj_ids_validation)], use_aug=False)
    dataset_test = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y_tone1": y_tone1_i, "y_tone2": y_tone2_i, "subj_id": subj_id_i,
    }) for X_i, y_tone1_i, y_tone2_i, subj_id_i in zip(Xs_test, ys_test_tone1, ys_test_tone2, subj_ids_test)], use_aug=False)
    # Shuffle and then batch the dataset.
    dataset_train = torch.utils.data.DataLoader(dataset_train,
        batch_size=params.train.batch_size, shuffle=True, drop_last=False)
    dataset_validation = torch.utils.data.DataLoader(dataset_validation,
        batch_size=params.train.batch_size, shuffle=True, drop_last=False)
    dataset_test = torch.utils.data.DataLoader(dataset_test,
        batch_size=params.train.batch_size, shuffle=True, drop_last=False)
    # Update related hyper-parameters in `params`.
    params.model.subj.n_subjects = params.model.n_subjects = n_subjects
    params.model.subj.d_input = params.model.n_channels = n_channels
    assert seq_len % params.model.seg_len == 0; params.model.seq_len = seq_len
    token_len = params.model.seq_len // params.model.tokenizer.seg_len
    params.model.tokenizer.token_len = token_len
    params.model.encoder.emb_len = token_len
    params.model.cls.d_feature = (
        params.model.encoder.d_model * params.model.encoder.emb_len
    )
    params.model.cls.n_labels = n_labels
    # Return the final `dataset_train` & `dataset_validation` & `dataset_test`.
    return dataset_train, dataset_validation, dataset_test

# def FinetuneDataset class
class FinetuneDataset(torch.utils.data.Dataset):
    """
    Brain signal finetune dataset for acoustic tone classification.
    """

    def __init__(self, data_items, use_aug=False, **kwargs):
        """
        Initialize `FinetuneDataset` object.

        Args:
            data_items: list - The list of data items, including [X,y_tone1,y_tone2,subj_id].
            use_aug: bool - The flag that indicates whether enable augmentations.
            kwargs: dict - The arguments related to initialize `torch.utils.data.Dataset`-style object.

        Returns:
            None
        """
        # First call super class init function to set up `torch.utils.data.Dataset`
        # style model and inherit it's functionality.
        super(FinetuneDataset, self).__init__(**kwargs)

        # Initialize parameters.
        self.data_items = data_items; self.use_aug = use_aug

        # Initialize variables.
        self._init_dataset()

    """
    init funcs
    """
    # def _init_dataset func
    def _init_dataset(self):
        """
        Initialize the configuration of dataset.

        Args:
            None

        Returns:
            None
        """
        # Initialize the maximum shift steps.
        self.max_steps = self.data_items[0].X.shape[1] // 10

    """
    dataset funcs
    """
    # def __len__ func
    def __len__(self):
        """
        Get the number of samples of dataset.

        Args:
            None

        Returns:
            n_samples: int - The number of samples of dataset.
        """
        return len(self.data_items)

    # def __getitem__ func
    def __getitem__(self, index):
        """
        Get the data item corresponding to data index.

        Args:
            index: int - The index of data item to get.

        Returns:
            data: dict - The data item dictionary.
        """
        ## Load data item.
        # Initialize `data_item` according to `index`.
        data_item = self.data_items[index]
        # Load data item from `data_item`.
        # X - (n_channels, seq_len); y_tone* - (n_tones,); subj_id - (n_subjects,)
        X = data_item.X; y_tone1 = data_item.y_tone1; y_tone2 = data_item.y_tone2; subj_id = data_item.subj_id
        ## Execute data augmentations.
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
        ## Construct the data dict.
        # Construct the final data dict.
        data = {
            "X": torch.from_numpy(X.T).to(dtype=torch.float32),
            "y_tone1": torch.from_numpy(y_tone1).to(dtype=torch.float32),
            "y_tone2": torch.from_numpy(y_tone2).to(dtype=torch.float32),
            "subj_id": torch.from_numpy(subj_id).to(dtype=torch.float32),
        }
        # Return the final `data`.
        return data

"""
train funcs
"""
# def train func
def train():
    """
    Train the model.

    Args:
        None

    Returns:
        None
    """
    global _forward, _train
    global params, paths, model, optimizer
    # Initialize the path of pretrained checkpoint.
    path_pt_ckpt = os.path.join(
        params.train.base, params.train.pt_ckpt
    ) if params.train.pt_ckpt is not None else None
    path_pt_params = os.path.join(
        params.train.base, *params.train.pt_ckpt.split(os.sep)[:-2], "save", "params"
    ) if params.train.pt_ckpt is not None else None
    # Load `n_subjects` & `n_channels` from `path_pt_params`.
    if path_pt_params is not None:
        params_pt = load_pickle(path_pt_params); n_subjects = params_pt.model.n_subjects; n_channels = params_pt.model.n_channels
    else:
        params_pt = None; n_subjects = None; n_channels = None
    # Log the start of current training process.
    paths.run.logger.summaries.info("Training started with dataset {}.".format(params.train.dataset))
    # Initialize model device.
    params.model.device = torch.device("cuda:{:d}".format(0)) if torch.cuda.is_available() else torch.device("cpu")
    print(params.model.device); paths.run.logger.summaries.info(params.model.device)
    # Initialize load_params. Each load_params_i corresponds to a sub-dataset.
    if params.train.dataset == "seeg_he2023xuanwu":
        # Initialize the configurations of subjects that we want to execute experiments.
        subjs_cfg = utils.DotDict({
            "001": utils.DotDict({
                "name": "001", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "001"),
                "ch_names": ["SM8", "SM9", "SM7", "SM11", "P4", "SM10", "SM6", "P3", "SM5", "CI9"],
            }),
            "002": utils.DotDict({
                "name": "002", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "002"),
                "ch_names": ["TI'2", "TI'3", "TI'1", "TI'6", "TI'4", "TI'7", "ST'3", "ST'2", "ST'4", "FP'4"],
            }),
            "003": utils.DotDict({
                "name": "003", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "003"),
                "ch_names": ["ST3", "ST1", "ST2", "ST9", "TI'4", "TI'3", "ST4", "TI'2", "ST7", "TI'8"] ,
            }),
            "004": utils.DotDict({
                "name": "004", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "004"),
                "ch_names": ["D12", "D13", "C4", "C3", "D11", "D14", "D10", "D9", "D5", "C15"],
            }),
            "005": utils.DotDict({
                "name": "005", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "005"),
                "ch_names": ["E8", "E9", "E6", "E7", "E11", "E12", "E5", "E10", "C10", "E4"],
            }),
            "006": utils.DotDict({
                "name": "006", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "006"),
                "ch_names": ["D3", "D1", "D6", "D2", "D5", "D4", "D7", "D8", "G8", "E13"],
            }),
            "007": utils.DotDict({
                "name": "007", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "007"),
                "ch_names": ["H2", "H4", "H3", "H1", "H6", "H5", "E4", "H7", "C13", "E5"],
            }),
            "008": utils.DotDict({
                "name": "008", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "008"),
                "ch_names": ["TI3", "TI4", "TI2", "TI5", "B9", "TI6", "TI7", "TI9", "TI10", "B5"],
            }),
            "009": utils.DotDict({
                "name": "009", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "009"),
                "ch_names": ["K9", "K8", "K6", "K7", "K11", "K10", "K5", "K4", "K3", "I9"],
            }),
            "010": utils.DotDict({
                "name": "010", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "010"),
                "ch_names": ["PI5", "PI6", "PI7", "PI8", "PI1", "PI9", "PI2", "SM2", "SP3", "PI4"],
            }),
            "011": utils.DotDict({
                "name": "011", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "011"),
                "ch_names": ["T2", "T3", "C9", "T4", "T5", "C7", "C8", "T1", "s1", "C4"],
            }),
            "012": utils.DotDict({
                "name": "012", "path": os.path.join(params.train.base, "data", "seeg.he2023xuanwu", "012"),
                "ch_names": ["TI'4", "TI'2", "TI'3", "TI'5", "TI'8", "TI'6", "TI'7", "TO'9", "P'5", "TO'8"],
            }),
        }); load_type = "bipolar_default"; load_task = "word_recitation"; use_align = False
        # Initialize the specified available_runs according to subjs_cfg.
        subjs_cfg = [subjs_cfg[subj_i] for subj_i in params.train.subjs]
        subj_idxs = params.train.subj_idxs; assert len(subj_idxs) == len(subjs_cfg)
        # Set `resample_rate` according to `load_type`.
        if load_type.startswith("bipolar"):
            resample_rate = 1000
        # `load_params` contains all the experiments that we want to execute for every run.
        load_params = [
            # train-task-all-speak-test-task-all-speak
            utils.DotDict({
                "name": "train-task-all-speak-test-task-all-speak", "type": load_type,
                "permutation": False, "resample_rate": resample_rate, "task": load_task, "use_align": use_align,
                "n_channels": n_channels, "n_subjects": n_subjects, "subj_idxs": subj_idxs,
            }),
        ]
    else:
        raise ValueError("ERROR: Unknown dataset {} in train.duin.run_acoustic_cls.".format(params.train.dataset))

    # Loop over all the experiments.
    for load_params_idx in range(len(load_params)):
        # Add `subjs_cfg` to `load_params_i`.
        load_params_i = cp.deepcopy(load_params[load_params_idx]); load_params_i.subjs_cfg = subjs_cfg
        # Log the start of current training iteration.
        msg = (
            "Training started with experiment {} with {:d} subjects."
        ).format(load_params_i.name, len(load_params_i.subjs_cfg))
        print(msg); paths.run.logger.summaries.info(msg)
        # Load data from specified experiment.
        dataset_train, dataset_validation, dataset_test = load_data(load_params_i)

        # Train the model for each time segment.
        accuracies_validation_tone1 = []; accuracies_validation_tone2 = []
        accuracies_test_tone1 = []; accuracies_test_tone2 = []

        # Reset the iteration information of params.
        params.iteration(iteration=0)
        # Initialize model of current time segment.
        model = duin_model(params.model)
        if path_pt_ckpt is not None: model.load_weight(path_pt_ckpt)
        model = model.to(device=params.model.device)
        if params.train.use_graph_mode: model = torch.compile(model)
        # Make an ADAM optimizer for model.
        optim_cfg = utils.DotDict({"name":"adamw","lr":params.train.lr_i,"weight_decay":0.05,})
        optimizer = utils.model.torch.create_optimizer(cfg=optim_cfg, model=model)

        for epoch_idx in range(params.train.n_epochs):
            # Update params according to `epoch_idx`, then update optimizer.lr.
            params.iteration(iteration=epoch_idx)
            for param_group_i in optimizer.param_groups: param_group_i["lr"] = params.train.lr_i
            # Record the start time of preparing data.
            time_start = time.time()
            # Prepare for model train process.
            accuracy_train_tone1 = []; accuracy_train_tone2 = []; loss_train = utils.DotDict()
            # Execute train process.
            for train_batch in dataset_train:
                # Initialize `batch_i` from `train_batch`.
                batch_i = [
                    train_batch["X"].to(device=params.model.device),
                    [train_batch["y_tone1"].to(device=params.model.device), train_batch["y_tone2"].to(device=params.model.device)],
                    train_batch["subj_id"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Train model for current batch.
                t_pred_i, loss_i = _train(batch_i)
                # Numpy the outputs of current batch.
                # t_pred_*_i: (batch_size, token_len, n_tones), t_true_*_i: (batch_size, n_tones)
                t_pred_tone1_i = t_pred_i[0].detach().cpu().numpy(); t_true_tone1_i = batch_i[1][0].detach().cpu().numpy()
                t_pred_tone2_i = t_pred_i[1].detach().cpu().numpy(); t_true_tone2_i = batch_i[1][1].detach().cpu().numpy()
                # Aggregate per-token predictions by averaging across token dimension
                # (batch_size, token_len, n_tones) -> (batch_size, n_tones)
                t_pred_tone1_i = np.mean(t_pred_tone1_i, axis=1)
                t_pred_tone2_i = np.mean(t_pred_tone2_i, axis=1)
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                accuracy_tone1_i = np.stack([
                    (np.argmax(t_pred_tone1_i, axis=-1) == np.argmax(t_true_tone1_i, axis=-1)).astype(np.int64),
                    np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
                ], axis=0).T; accuracy_train_tone1.append(accuracy_tone1_i)
                accuracy_tone2_i = np.stack([
                    (np.argmax(t_pred_tone2_i, axis=-1) == np.argmax(t_true_tone2_i, axis=-1)).astype(np.int64),
                    np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
                ], axis=0).T; accuracy_train_tone2.append(accuracy_tone2_i)
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_train, key_i):
                        loss_train[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_train[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
            # Record information related to train process.
            accuracy_train_tone1 = np.concatenate(accuracy_train_tone1, axis=0)
            accuracy_train_tone1 = np.array([accuracy_train_tone1[np.where(accuracy_train_tone1[:,1] == subj_idx),0].mean()\
                for subj_idx in sorted(set(accuracy_train_tone1[:,1]))], dtype=np.float32)
            accuracy_train_tone2 = np.concatenate(accuracy_train_tone2, axis=0)
            accuracy_train_tone2 = np.array([accuracy_train_tone2[np.where(accuracy_train_tone2[:,1] == subj_idx),0].mean()\
                for subj_idx in sorted(set(accuracy_train_tone2[:,1]))], dtype=np.float32)
            for key_i, item_i in loss_train.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_train[key_i] = item_i

            # Prepare for model validation process.
            accuracy_validation_tone1 = []; accuracy_validation_tone2 = []; loss_validation = utils.DotDict()
            # Execute validation process.
            for validation_batch in dataset_validation:
                # Initialize `batch_i` from `validation_batch`.
                batch_i = [
                    validation_batch["X"].to(device=params.model.device),
                    [validation_batch["y_tone1"].to(device=params.model.device), validation_batch["y_tone2"].to(device=params.model.device)],
                    validation_batch["subj_id"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Validate model for current batch.
                t_pred_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                # t_pred_*_i: (batch_size, token_len, n_tones), t_true_*_i: (batch_size, n_tones)
                t_pred_tone1_i = t_pred_i[0].detach().cpu().numpy(); t_true_tone1_i = batch_i[1][0].detach().cpu().numpy()
                t_pred_tone2_i = t_pred_i[1].detach().cpu().numpy(); t_true_tone2_i = batch_i[1][1].detach().cpu().numpy()
                # Aggregate per-token predictions by averaging across token dimension
                # (batch_size, token_len, n_tones) -> (batch_size, n_tones)
                t_pred_tone1_i = np.mean(t_pred_tone1_i, axis=1)
                t_pred_tone2_i = np.mean(t_pred_tone2_i, axis=1)
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                accuracy_tone1_i = np.stack([
                    (np.argmax(t_pred_tone1_i, axis=-1) == np.argmax(t_true_tone1_i, axis=-1)).astype(np.int64),
                    np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
                ], axis=0).T; accuracy_validation_tone1.append(accuracy_tone1_i)
                accuracy_tone2_i = np.stack([
                    (np.argmax(t_pred_tone2_i, axis=-1) == np.argmax(t_true_tone2_i, axis=-1)).astype(np.int64),
                    np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
                ], axis=0).T; accuracy_validation_tone2.append(accuracy_tone2_i)
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_validation, key_i):
                        loss_validation[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_validation[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
            # Record information related to validation process.
            accuracy_validation_tone1 = np.concatenate(accuracy_validation_tone1, axis=0)
            accuracy_validation_tone1 = np.array([accuracy_validation_tone1[np.where(accuracy_validation_tone1[:,1] == subj_idx),0].mean()\
                for subj_idx in sorted(set(accuracy_validation_tone1[:,1]))], dtype=np.float32)
            accuracy_validation_tone2 = np.concatenate(accuracy_validation_tone2, axis=0)
            accuracy_validation_tone2 = np.array([accuracy_validation_tone2[np.where(accuracy_validation_tone2[:,1] == subj_idx),0].mean()\
                for subj_idx in sorted(set(accuracy_validation_tone2[:,1]))], dtype=np.float32)
            for key_i, item_i in loss_validation.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_validation[key_i] = item_i
            accuracies_validation_tone1.append(accuracy_validation_tone1)
            accuracies_validation_tone2.append(accuracy_validation_tone2)

            # Prepare for model test process.
            accuracy_test_tone1 = []; accuracy_test_tone2 = []; loss_test = utils.DotDict()
            # Execute test process.
            for test_batch in dataset_test:
                # Initialize `batch_i` from `test_batch`.
                batch_i = [
                    test_batch["X"].to(device=params.model.device),
                    [test_batch["y_tone1"].to(device=params.model.device), test_batch["y_tone2"].to(device=params.model.device)],
                    test_batch["subj_id"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Test model for current batch.
                t_pred_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                # t_pred_*_i: (batch_size, token_len, n_tones), t_true_*_i: (batch_size, n_tones)
                t_pred_tone1_i = t_pred_i[0].detach().cpu().numpy(); t_true_tone1_i = batch_i[1][0].detach().cpu().numpy()
                t_pred_tone2_i = t_pred_i[1].detach().cpu().numpy(); t_true_tone2_i = batch_i[1][1].detach().cpu().numpy()
                # Aggregate per-token predictions by averaging across token dimension
                # (batch_size, token_len, n_tones) -> (batch_size, n_tones)
                t_pred_tone1_i = np.mean(t_pred_tone1_i, axis=1)
                t_pred_tone2_i = np.mean(t_pred_tone2_i, axis=1)
                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                accuracy_tone1_i = np.stack([
                    (np.argmax(t_pred_tone1_i, axis=-1) == np.argmax(t_true_tone1_i, axis=-1)).astype(np.int64),
                    np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
                ], axis=0).T; accuracy_test_tone1.append(accuracy_tone1_i)
                accuracy_tone2_i = np.stack([
                    (np.argmax(t_pred_tone2_i, axis=-1) == np.argmax(t_true_tone2_i, axis=-1)).astype(np.int64),
                    np.argmax(batch_i[2].detach().cpu().numpy(), axis=-1).astype(np.int64),
                ], axis=0).T; accuracy_test_tone2.append(accuracy_tone2_i)
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_test, key_i):
                        loss_test[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_test[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
            # Record information related to test process.
            accuracy_test_tone1 = np.concatenate(accuracy_test_tone1, axis=0)
            accuracy_test_tone1 = np.array([accuracy_test_tone1[np.where(accuracy_test_tone1[:,1] == subj_idx),0].mean()\
                for subj_idx in sorted(set(accuracy_test_tone1[:,1]))], dtype=np.float32)
            accuracy_test_tone2 = np.concatenate(accuracy_test_tone2, axis=0)
            accuracy_test_tone2 = np.array([accuracy_test_tone2[np.where(accuracy_test_tone2[:,1] == subj_idx),0].mean()\
                for subj_idx in sorted(set(accuracy_test_tone2[:,1]))], dtype=np.float32)
            for key_i, item_i in loss_test.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_test[key_i] = item_i
            accuracies_test_tone1.append(accuracy_test_tone1)
            accuracies_test_tone2.append(accuracy_test_tone2)

            ## Write progress to summaries.
            # Log information related to current training epoch.
            time_stop = time.time()
            msg = (
                "Finish train epoch {:d} in {:.2f} seconds."
            ).format(epoch_idx, time_stop-time_start)
            print(msg); paths.run.logger.summaries.info(msg)
            # Log information related to train process.
            msg = "Accuracy(train-tone1): [{:.2f}%".format(accuracy_train_tone1[0] * 100.)
            for subj_idx in range(1, len(accuracy_train_tone1)): msg += ",{:.2f}%".format(accuracy_train_tone1[subj_idx] * 100.)
            msg += "]; Accuracy(train-tone2): [{:.2f}%".format(accuracy_train_tone2[0] * 100.)
            for subj_idx in range(1, len(accuracy_train_tone2)): msg += ",{:.2f}%".format(accuracy_train_tone2[subj_idx] * 100.)
            msg += "].\n"; loss_keys = list(loss_train.keys())
            msg += "Loss(train): {:.5f} ({})".format(loss_train[loss_keys[0]], loss_keys[0])
            for loss_idx in range(1, len(loss_keys)):
                msg += "; {:.5f} ({})".format(loss_train[loss_keys[loss_idx]], loss_keys[loss_idx])
            print(msg); paths.run.logger.summaries.info(msg)
            # Log information related to validation process.
            msg = "Accuracy(validation-tone1): [{:.2f}%".format(accuracy_validation_tone1[0] * 100.)
            for subj_idx in range(1, len(accuracy_validation_tone1)): msg += ",{:.2f}%".format(accuracy_validation_tone1[subj_idx] * 100.)
            msg += "]; Accuracy(validation-tone2): [{:.2f}%".format(accuracy_validation_tone2[0] * 100.)
            for subj_idx in range(1, len(accuracy_validation_tone2)): msg += ",{:.2f}%".format(accuracy_validation_tone2[subj_idx] * 100.)
            msg += "].\n"; loss_keys = list(loss_validation.keys())
            msg += "Loss(validation): {:.5f} ({})".format(loss_validation[loss_keys[0]], loss_keys[0])
            for loss_idx in range(1, len(loss_keys)):
                msg += "; {:.5f} ({})".format(loss_validation[loss_keys[loss_idx]], loss_keys[loss_idx])
            print(msg); paths.run.logger.summaries.info(msg)
            # Log information related to test process.
            msg = "Accuracy(test-tone1): [{:.2f}%".format(accuracy_test_tone1[0] * 100.)
            for subj_idx in range(1, len(accuracy_test_tone1)): msg += ",{:.2f}%".format(accuracy_test_tone1[subj_idx] * 100.)
            msg += "]; Accuracy(test-tone2): [{:.2f}%".format(accuracy_test_tone2[0] * 100.)
            for subj_idx in range(1, len(accuracy_test_tone2)): msg += ",{:.2f}%".format(accuracy_test_tone2[subj_idx] * 100.)
            msg += "].\n"; loss_keys = list(loss_test.keys())
            msg += "Loss(test): {:.5f} ({})".format(loss_test[loss_keys[0]], loss_keys[0])
            for loss_idx in range(1, len(loss_keys)):
                msg += "; {:.5f} ({})".format(loss_test[loss_keys[loss_idx]], loss_keys[loss_idx])
            print(msg); paths.run.logger.summaries.info(msg)
            ## Write progress to tensorboard.
            # Get the pointer of writer.
            writer = paths.run.logger.tensorboard
            # Log information related to train process.
            for key_i, loss_i in loss_train.items():
                writer.add_scalar(os.path.join("losses", "train", key_i), loss_i, global_step=epoch_idx)
            for subj_idx, accuracy_i in enumerate(accuracy_train_tone1):
                subj_i = load_params_i.subjs_cfg[subj_idx].name
                writer.add_scalar(os.path.join("accuracies", "train-tone1", subj_i), accuracy_i, global_step=epoch_idx)
            for subj_idx, accuracy_i in enumerate(accuracy_train_tone2):
                subj_i = load_params_i.subjs_cfg[subj_idx].name
                writer.add_scalar(os.path.join("accuracies", "train-tone2", subj_i), accuracy_i, global_step=epoch_idx)
            # Log information related to validation process.
            for key_i, loss_i in loss_validation.items():
                writer.add_scalar(os.path.join("losses", "validation", key_i), loss_i, global_step=epoch_idx)
            for subj_idx, accuracy_i in enumerate(accuracy_validation_tone1):
                subj_i = load_params_i.subjs_cfg[subj_idx].name
                writer.add_scalar(os.path.join("accuracies", "validation-tone1", subj_i), accuracy_i, global_step=epoch_idx)
            for subj_idx, accuracy_i in enumerate(accuracy_validation_tone2):
                subj_i = load_params_i.subjs_cfg[subj_idx].name
                writer.add_scalar(os.path.join("accuracies", "validation-tone2", subj_i), accuracy_i, global_step=epoch_idx)
            # Log information related to test process.
            for key_i, loss_i in loss_test.items():
                writer.add_scalar(os.path.join("losses", "test", key_i), loss_i, global_step=epoch_idx)
            for subj_idx, accuracy_i in enumerate(accuracy_test_tone1):
                subj_i = load_params_i.subjs_cfg[subj_idx].name
                writer.add_scalar(os.path.join("accuracies", "test-tone1", subj_i), accuracy_i, global_step=epoch_idx)
            for subj_idx, accuracy_i in enumerate(accuracy_test_tone2):
                subj_i = load_params_i.subjs_cfg[subj_idx].name
                writer.add_scalar(os.path.join("accuracies", "test-tone2", subj_i), accuracy_i, global_step=epoch_idx)
            # Summarize model information.
            if epoch_idx == 0:
                msg = summary(model, col_names=("num_params", "params_percent", "trainable",))
                print(msg); paths.run.logger.summaries.info(msg)

        # Log information related to channel weights.
        ch_weights = model.get_weight_i().numpy()
        for subj_idx, subj_cfg_i in enumerate(load_params_i.subjs_cfg):
            ch_names_i = subj_cfg_i.ch_names; ch_weights_i = ch_weights[subj_idx,...]
            assert len(ch_weights_i.shape) == 1; ch_weights_i = ch_weights_i[:len(ch_names_i)]
            ch_orders_i = np.argsort(ch_weights_i)[::-1]
            top_k = min(10, len(ch_names_i)); msg = (
                "INFO: The top-{:d} channels are {} with weights {}."
            ).format(top_k, [ch_names_i[ch_orders_i[top_idx]] for top_idx in range(top_k)],
                [ch_weights_i[ch_orders_i[top_idx]] for top_idx in range(top_k)])
            print(msg); paths.run.logger.summaries.info(msg)

        # Convert accuracies to numpy arrays
        accuracies_validation_tone1 = np.round(np.array(accuracies_validation_tone1, dtype=np.float32), decimals=4).T
        accuracies_validation_tone2 = np.round(np.array(accuracies_validation_tone2, dtype=np.float32), decimals=4).T
        accuracies_test_tone1 = np.round(np.array(accuracies_test_tone1, dtype=np.float32), decimals=4).T
        accuracies_test_tone2 = np.round(np.array(accuracies_test_tone2, dtype=np.float32), decimals=4).T

        # Calculate average validation accuracy for selecting best epoch
        accuracies_validation_avg = (accuracies_validation_tone1 + accuracies_validation_tone2) / 2.0
        accuracies_test_avg = (accuracies_test_tone1 + accuracies_test_tone2) / 2.0

        # Find epoch with maximum average validation accuracy
        epoch_maxacc_idxs = [np.where(
            accuracies_validation_avg[subj_idx] == np.max(accuracies_validation_avg[subj_idx])
        )[0] for subj_idx in range(accuracies_validation_avg.shape[0])]
        epoch_maxacc_idxs = [epoch_maxacc_idxs[subj_idx][
            np.argmax(accuracies_test_avg[subj_idx,epoch_maxacc_idxs[subj_idx]])
        ] for subj_idx in range(len(epoch_maxacc_idxs))]

        # Finish training process of current specified experiment.
        msg = (
            "Finish the training process of experiment {}."
        ).format(load_params_i.name)
        print(msg); paths.run.logger.summaries.info(msg)
        assert len(load_params_i.subjs_cfg) == len(epoch_maxacc_idxs)
        for subj_idx in range(len(load_params_i.subjs_cfg)):
            subj_i = load_params_i.subjs_cfg[subj_idx].name; epoch_maxacc_idx_i = epoch_maxacc_idxs[subj_idx]
            accuracy_validation_tone1_i = accuracies_validation_tone1[subj_idx,epoch_maxacc_idx_i]
            accuracy_validation_tone2_i = accuracies_validation_tone2[subj_idx,epoch_maxacc_idx_i]
            accuracy_test_tone1_i = accuracies_test_tone1[subj_idx,epoch_maxacc_idx_i]
            accuracy_test_tone2_i = accuracies_test_tone2[subj_idx,epoch_maxacc_idx_i]
            msg = (
                "For subject {}, we get test-accuracy tone1({:.2f}%) tone2({:.2f}%) according to max validation-accuracy tone1({:.2f}%) tone2({:.2f}%) at epoch {:d}."
            ).format(subj_i, accuracy_test_tone1_i * 100., accuracy_test_tone2_i * 100.,
                     accuracy_validation_tone1_i * 100., accuracy_validation_tone2_i * 100., epoch_maxacc_idx_i)
            print(msg); paths.run.logger.summaries.info(msg)

    # Finish current training process.
    writer = paths.run.logger.tensorboard; writer.close()
    # Log the end of current training process.
    msg = "Training finished with dataset {}.".format(params.train.dataset)
    print(msg); paths.run.logger.summaries.info(msg)

# def _forward func
def _forward(inputs):
    """
    Forward the model using one-step data. Everything entering this function already be a tensor.

    Args:
        inputs: tuple - The input data, including (X, [y_tone1_true, y_tone2_true], subj_id).

    Returns:
        t_pred: list - The predicted tones [tone1_pred, tone2_pred].
        loss: DotDict - The loss dictionary.
    """
    global model; model.eval()
    with torch.no_grad():
        # Note: Need to add token_mask
        token_mask = torch.ones((inputs[0].shape[0], 1), dtype=inputs[0].dtype, device=inputs[0].device)
        inputs_with_mask = [inputs[0], inputs[1], inputs[2], token_mask]
        return model(inputs_with_mask)

# def _train func
def _train(inputs):
    """
    Train the model using one-step data. Everything entering this function already be a tensor.

    Args:
        inputs: tuple - The input data, including (X, [y_tone1_true, y_tone2_true], subj_id).

    Returns:
        t_pred: list - The predicted tones [tone1_pred, tone2_pred].
        loss: DotDict - The loss dictionary.
    """
    global model, optimizer; model.train()
    # Add token_mask to inputs
    token_mask = torch.ones((inputs[0].shape[0], 1), dtype=inputs[0].dtype, device=inputs[0].device)
    inputs_with_mask = [inputs[0], inputs[1], inputs[2], token_mask]
    # Forward model to get the corresponding loss.
    t_pred, loss = model(inputs_with_mask)
    # Use optimizer to update parameters.
    optimizer.zero_grad(); loss["total"].backward(); optimizer.step()
    # Return the final `t_pred` & `loss`.
    return t_pred, loss

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
        parser: object - The initialized argument parser.
    """
    # Initialize parser.
    parser = argparse.ArgumentParser("DuIN Acoustic Tone CLS for brain signals", add_help=False)

    # Basic training parameters
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,])
    parser.add_argument("--subjs", type=str, nargs="+", default=["001",])
    parser.add_argument("--subj_idxs", type=int, nargs="+", default=[0,])
    parser.add_argument("--pt_ckpt", type=str, default=None)

    # Learning rate schedule
    parser.add_argument("--lr_min", type=float, default=1e-5, help="Minimum learning rate (default: 1e-5)")
    parser.add_argument("--lr_max", type=float, default=5e-4, help="Maximum learning rate (default: 5e-4)")

    # Training schedule
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument("--warmup_epochs", type=int, default=20, help="Number of warmup epochs (default: 20)")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size (default: 64)")

    # Encoder dropout rates
    parser.add_argument("--attn_dropout", type=float, default=0.1, help="Attention dropout rate (default: 0.1)")
    parser.add_argument("--ff_dropout", type=str, default="0.1,0.0", help="Feedforward dropout rates, comma-separated (default: '0.1,0.0')")

    # Encoder architecture
    parser.add_argument("--n_blocks", type=int, default=8, help="Number of transformer blocks (default: 8)")
    parser.add_argument("--n_heads", type=int, default=8, help="Number of attention heads (default: 8)")

    # Classification head parameters
    parser.add_argument("--d_hidden", type=str, default="", help="Hidden dimensions for classification head, comma-separated (default: '' - no hidden layers)")
    parser.add_argument("--cls_dropout", type=float, default=0.5, help="Dropout rate for classification head (default: 0.5)")

    # Shell script path for saving configuration
    parser.add_argument("--run_script", type=str, default=None, help="Path to the shell script used to launch training (will be saved to summaries folder)")

    # Return the final `parser`.
    return parser

if __name__ == "__main__":
    import os, shutil
    # local dep
    from params.duin_params import duin_acoustic_cls_params as duin_params

    # macro
    dataset = "seeg_he2023xuanwu"

    # Initialize base path.
    base = os.path.join(os.getcwd(), os.pardir, os.pardir)
    # Initialize arguments parser.
    args_parser = get_args_parser(); args = args_parser.parse_args()
    # Initialize duin_params.
    duin_params_inst = duin_params(dataset=dataset)
    duin_params_inst.train.base = base; duin_params_inst.train.subjs = args.subjs
    duin_params_inst.train.subj_idxs = args.subj_idxs; duin_params_inst.train.pt_ckpt = args.pt_ckpt

    # Apply hyperparameters from command line arguments
    # Learning rate schedule
    duin_params_inst.train.lr_min = args.lr_min
    duin_params_inst.train.lr_max = args.lr_max
    # Training schedule
    duin_params_inst.train.n_epochs = args.n_epochs
    duin_params_inst.train.warmup_epochs = args.warmup_epochs
    duin_params_inst.train.batch_size = args.batch_size
    # Encoder dropout rates
    duin_params_inst.model.encoder.attn_dropout = args.attn_dropout
    ff_dropout_list = [float(x) for x in args.ff_dropout.split(",")]
    duin_params_inst.model.encoder.ff_dropout = ff_dropout_list
    # Encoder architecture
    duin_params_inst.model.encoder.n_blocks = args.n_blocks
    duin_params_inst.model.encoder.n_heads = args.n_heads
    # Classification head parameters
    if args.d_hidden:
        d_hidden_list = [int(x) for x in args.d_hidden.split(",")]
        duin_params_inst.model.cls.d_hidden = d_hidden_list
    duin_params_inst.model.cls.dropout = args.cls_dropout

    # Initialize the training process.
    init(duin_params_inst)

    # Save the run script to summaries folder if provided
    if args.run_script is not None and os.path.exists(args.run_script):
        try:
            shutil.copy(args.run_script, os.path.join(paths.run.save, "run_script.sh"))
            print(f"[INFO] Saved run script to {os.path.join(paths.run.save, 'run_script.sh')}")
        except Exception as e:
            print(f"[WARNING] Failed to save run script: {e}")

    # Loop the training process over random seeds.
    for seed_i in args.seeds:
        # Initialize random seed, then train duin.
        utils.model.torch.set_seeds(seed_i); train()
