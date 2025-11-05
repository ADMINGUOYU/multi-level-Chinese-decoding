#!/usr/bin/env python3
"""
Created on 22:45, Jan. 21st, 2024

@author: Norbert Zheng
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
    # sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))
import utils; import utils.model.torch; import utils.data.seeg
from utils.data import load_pickle
from models.duin import duin_align as duin_model
# GPU DEBUGGING: Disable cuDNN to test if it causes GPU training failure
import torch.backends.cudnn as cudnn
cudnn.enabled = False
cudnn.benchmark = False
cudnn.deterministic = True
print("WARNING: cuDNN DISABLED for debuggin GPU training issue")


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
def init(params_):   # 不变
    """
    Initialize `duin_cls` training variables.

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

# def _init_model func
def _init_model():   # 不变
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
def _init_train():   # 不变
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
def load_data(load_params):   # 不变
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
            "ERROR: Unknown dataset type {} in train.duin.run_cls."
        ).format(params.train.dataset))
    # Return the final `dataset_train` & `dataset_validation` & `dataset_test`.
    return dataset_train, dataset_validation, dataset_test

# def _load_data_seeg_he2023xuanwu func   # 大修
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
    print("DEBUG params.model keys:", list(params.model.keys()))

    # --- Load supervision embeddings ---
    emb_path = "/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Visual_GT_VitPerchar.npz"   # 新增加载监督 embedding 表
    emb_data = np.load(emb_path)
    emb_table = emb_data["emb_mean"] 
    emb_words = emb_data["words"]  # 或 "chars"，看 npz 内部字段名
    emb_table = emb_table / (np.linalg.norm(emb_table, axis=1, keepdims=True) + 1e-8)    # Normalize embeddings 
    print("Embedding table shape:", emb_table.shape)
    print("[CHECK] emb_table nan count:", np.isnan(emb_table).sum())

    # ===== 关键修正开始：用“data”pickle 收集真实的类别名顺序 =====
    subjs_cfg = load_params.subjs_cfg

    import unicodedata, pickle
    def _norm_text(s: str) -> str:
        # 统一文本的 Unicode 规格 & 去除首尾空白，避免“外卖 ”、“外\u3000卖”之类的细微差异
        return unicodedata.normalize("NFKC", str(s)).strip()
    # 从第一个受试者的所有 run 中，读取 data.pkl，收集出现过的 label 名（sample.name）
    # 用 union + 排序的方式获得与 _load_subj_bipolar 中一致的顺序（那里也是 sorted(set(...))）
    label_name_set = set()
    subj0_path = subjs_cfg[0].path
    task_dir = os.path.join(subj0_path, "word-recitation")
    dataset_dir_name = f"dataset.bipolar.default.{ 'aligned' if load_params.use_align else 'unaligned' }"
    for run_i in os.listdir(task_dir):
        data_pkl = os.path.join(task_dir, run_i, dataset_dir_name, "data")
        if not os.path.exists(data_pkl):
            continue
        dataset_data_i = load_pickle(data_pkl)  # 列表，元素有 .name / .data_s / .data_n 等属性
        for sample in dataset_data_i:
            if hasattr(sample, "name"):
                label_name_set.add(_norm_text(sample.name))
    # 得到“按名字排序”的类别顺序，这与 _load_subj_bipolar 里构造 y 的做法一致
    label_order = sorted(label_name_set)

    print(f"[DEBUG] label_order size={len(label_order)}, head={label_order[:61]}")

    # 把 embedding 表按 label_order 重新排列
    # 先把 emb_words 规范化后建立索引
    emb_words_norm = [_norm_text(w) for w in emb_words]
    word_to_idx = {w: i for i, w in enumerate(emb_words_norm)}

    aligned_emb_table = np.zeros((len(label_order), emb_table.shape[1]), dtype=emb_table.dtype)
    missing = []
    for i, name in enumerate(label_order):
        j = word_to_idx.get(_norm_text(name), None)
        if j is None:
            missing.append(name)
            # 没找到就用 0 向量占位（或 np.random.randn，建议先 0 以便肉眼发现问题）
            aligned_emb_table[i, :] = 0.0
        else:
            aligned_emb_table[i, :] = emb_table[j, :]
    if missing:
        print(f"[WARNING] {len(missing)} labels not found in embedding table. e.g. {missing[:10]}")

    print("---- [CHECK ALIGNMENT] ----")
    for i in range(10):
        label = label_order[i]
        match = label if label in emb_words else "❌ NOT FOUND"
        print(f"Dataset label[{i}]: {label:>6s}  →  Embedding match: {match}")

    emb_table = aligned_emb_table    # 用重新对齐后的表覆盖
    emb_words = np.array(label_order)   # 同步更新词序
    print("[INFO] Embedding table reordered to match dataset label names.")
    params.model.emb_table = emb_table
    params.model.emb_words = np.array(label_order)  

    # Initialize subjs_cfg.
    subjs_cfg = load_params.subjs_cfg    # 读取实验配置中定义的受试者信息
    # Initialize `n_subjects` & `n_subjects` & `subj_idxs` & `seq_len` & `n_labels` from `load_params`.
    n_channels = load_params.n_channels if load_params.n_channels is not None else None
    n_subjects = load_params.n_subjects if load_params.n_subjects is not None else len(subjs_cfg)
    subj_idxs = load_params.subj_idxs if load_params.subj_idxs is not None else np.arange(n_subjects)
    seq_len = None; n_labels = None
    # Initialize `Xs_*` & `ys_*` & `subj_ids_*`, then load them.
    Xs_train = []; ys_train = []; subj_ids_train = []
    Xs_validation = []; ys_validation = []; subj_ids_validation = []
    Xs_test = []; ys_test = []; subj_ids_test = []

    for subj_idx, subj_cfg_i in zip(subj_idxs, subjs_cfg):
        # Load data from specified subject run.
        func = getattr(getattr(utils.data.seeg.he2023xuanwu, load_params.task), "load_subj_{}".format(load_params.type))   # 遍历每个 subject 加载原始信号
        dataset = func(subj_cfg_i.path, ch_names=subj_cfg_i.ch_names, use_align=load_params.use_align)
        X = dataset.X_s.astype(np.float32); y = dataset.y.astype(np.int64)
        # If the type of dataset is `bipolar`.
        if load_params.type.startswith("bipolar"):
            # Truncate `X` to let them have the same length.
            # TODO: Here, we only keep the [0.0~0.8]s-part of [audio,image] that after onset. And we should
            # note that the [0.0~0.8]s-part of image is the whole onset time of image, the [0.0~0.8]s-part
            # of audio is the sum of the whole onset time of audio and the following 0.3s padding.
            # X - (n_samples, seq_len, n_channels)
            X = X
            # Resample the original data to the specified `resample_rate`.
            sample_rate = 1000; X = sp.signal.resample(X, int(np.round(X.shape[1] /\
                (sample_rate / load_params.resample_rate))), axis=1)
            # Truncate data according to epoch range (-0.2,1.0), the original epoch range is (-0.5,2.0).
            X = X[:,int(np.round((-0.5 - (-0.5)) * load_params.resample_rate)):\
                int(np.round((2.5 - (-0.5)) * load_params.resample_rate)),:]
            # Do Z-score for each channel.
            # TODO: As we do z-score for each channel, we do not have to scale the reconstruction
            # loss by the variance of each channel. We can check `np.var(X, axis=(0,1))` is near 1.
            X = (X - np.mean(X, axis=(0,1), keepdims=True)) / np.std(X, axis=(0,1), keepdims=True)
        # Get unknown type of dataset.
        else:
            raise ValueError("ERROR: Unknown type {} of dataset.".format(load_params.type))
        # Initialize trainset & testset.
        # X - (n_samples, seq_len, n_channels); y - (n_samples,)
        train_ratio = params.train.train_ratio; train_idxs = []; test_idxs = []

        for label_i in sorted(set(y)):   # 训练 / 测试划分
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

        # === Debug: check label-embedding correspondence ===
        emb_table = getattr(params.model, "emb_table", emb_table)
        emb_words = getattr(params.model, "emb_words", emb_words)

        print("[DEBUG] Subject:", subj_cfg_i.name)
        print("→ Dataset unique labels:", labels)
        print("→ Embedding table words (前10个):", emb_words[:10])
        print("→ Embedding table size:", emb_table.shape)

        # Convert class indices to embedding supervision
        y_train_idx = np.array([labels.index(y_i) for y_i in y_train], dtype=np.int64)   # 删除原有的 one-hot 生成，改为根据标签索引提取对应的 768 维 embedding
        y_test_idx = np.array([labels.index(y_i) for y_i in y_test], dtype=np.int64)
        y_train = emb_table[y_train_idx]  # (n_train, 768)
        y_test = emb_table[y_test_idx]    # (n_test, 768)
        print(y_train.shape)

        # Execute sample permutation. We only shuffle along the axis.
        if load_params.permutation: np.random.shuffle(y_train)
        # Further split test-set into validation-set & test-set.
        validation_idxs = np.random.choice(np.arange(X_test.shape[0]), size=int(X_test.shape[0]/2), replace=False)
        validation_mask = np.zeros((X_test.shape[0],), dtype=np.bool_); validation_mask[validation_idxs] = True
        X_validation = X_test[validation_mask,:,:]; y_validation = y_test[validation_mask,:]
        X_test = X_test[~validation_mask,:,:]; y_test = y_test[~validation_mask,:]
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
        Xs_train.append(X_train); ys_train.append(y_train); subj_ids_train.append(subj_id_train)   # 把所有受试者拼在一起
        Xs_validation.append(X_validation); ys_validation.append(y_validation); subj_ids_validation.append(subj_id_validation)
        Xs_test.append(X_test); ys_test.append(y_test); subj_ids_test.append(subj_id_test)
        # Update `n_channels` & `seq_len` & `n_labels`.
        n_channels = max(X.shape[-1], n_channels) if n_channels is not None else X.shape[-1]
        seq_len = X.shape[-2] if seq_len is None else seq_len; assert seq_len == X.shape[-2]
        n_labels = len(labels) if n_labels is None else n_labels; assert n_labels == len(labels)

    # Check `n_channels` according to `load_params`.
    if load_params.n_channels is not None: assert n_channels == load_params.n_channels
    # Update `Xs_*` with `n_channels`.
    # TODO: We pad 0s to solve the problem that different subjects have different number of channels.
    # Thus we can use one `Dense` layer in the subject layer to get the corresponding mapping matrix.
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
    Xs_train = np.concatenate(Xs_train, axis=0); ys_train = np.concatenate(ys_train, axis=0)
    subj_ids_train = np.concatenate(subj_ids_train, axis=0)
    Xs_validation = np.concatenate(Xs_validation, axis=0); ys_validation = np.concatenate(ys_validation, axis=0)
    subj_ids_validation = np.concatenate(subj_ids_validation, axis=0)
    Xs_test = np.concatenate(Xs_test, axis=0); ys_test = np.concatenate(ys_test, axis=0)
    subj_ids_test = np.concatenate(subj_ids_test, axis=0)
    # Shuffle dataset to fuse different subjects.
    train_idxs = np.arange(Xs_train.shape[0]); np.random.shuffle(train_idxs)
    validation_idxs = np.arange(Xs_validation.shape[0]); np.random.shuffle(validation_idxs)
    test_idxs = np.arange(Xs_test.shape[0]); np.random.shuffle(test_idxs)
    Xs_train = Xs_train[train_idxs,...]; ys_train = ys_train[train_idxs,...]; subj_ids_train = subj_ids_train[train_idxs,...]
    Xs_validation = Xs_validation[validation_idxs,...]; ys_validation = ys_validation[validation_idxs,...]
    subj_ids_validation = subj_ids_validation[validation_idxs,...]
    Xs_test = Xs_test[test_idxs,...]; ys_test = ys_test[test_idxs,...]; subj_ids_test = subj_ids_test[test_idxs,...]
    # Log information of data loading.
    msg = (
        "INFO: Data preparation complete, with train-set ({}) & validation-set ({}) & test-set ({})."
    ).format(Xs_train.shape, Xs_validation.shape, Xs_test.shape)
    print(msg); paths.run.logger.summaries.info(msg)

    # Construct dataset from data items.
    dataset_train = FinetuneDataset(data_items=[utils.DotDict({   # 构建 PyTorch Dataset + Dataloader
        "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
    }) for X_i, y_i, subj_id_i in zip(Xs_train, ys_train, subj_ids_train)], use_aug=True)
    dataset_validation = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
    }) for X_i, y_i, subj_id_i in zip(Xs_validation, ys_validation, subj_ids_validation)], use_aug=False)
    dataset_test = FinetuneDataset(data_items=[utils.DotDict({
        "X": X_i.T, "y": y_i, "subj_id": subj_id_i,
    }) for X_i, y_i, subj_id_i in zip(Xs_test, ys_test, subj_ids_test)], use_aug=False)
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

    params.model.align.d_feature = (
        params.model.encoder.d_model * params.model.encoder.emb_len
    )
    params.model.align.d_target = emb_table.shape[1]  # 768

    # Return the final `dataset_train` & `dataset_validation` & `dataset_test`.
    return dataset_train, dataset_validation, dataset_test

# def FinetuneDataset class
class FinetuneDataset(torch.utils.data.Dataset):   # 不变
    """
    Brain signal finetune dataset.
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
        # X - (n_channels, seq_len); y - (n_labels,); subj_id - (n_subjects,)
        X = data_item.X; y = data_item.y; subj_id = data_item.subj_id
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
            "y": torch.from_numpy(y).to(dtype=torch.float32),
            "subj_id": torch.from_numpy(subj_id).to(dtype=torch.float32),
        }
        # Return the final `data`.
        return data

"""
train funcs
"""
# def train func
def train():   # 修改训练循环中的损失函数、输出日志
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
    elif params.train.dataset == "eeg_zhou2023cibr":
        # Initialize the configurations of subjects that we want to execute experiments.
        subjs_cfg = [
            #utils.DotDict({
            #    "name": "021", "path": os.path.join(paths.base, "data", "eeg.zhou2023cibr", "021", "20230407"),
            #}),
            utils.DotDict({
                "name": "023", "path": os.path.join(paths.base, "data", "eeg.zhou2023cibr", "023", "20230412"),
            }),
        ]; load_type = "default"
        # Initialize the specified available_runs according to subjs_cfg.
        subj_idxs = [0,]; assert len(subj_idxs) == len(subjs_cfg)
        # `load_params` contains all the experiments that we want to execute for every run.
        load_params = [
            # train-task-all-image-test-task-all-image
            utils.DotDict({
                "name": "train-task-all-image-test-task-all-image",
                "trainset": [
                    "task-image-audio-pre-image", "task-audio-image-pre-image",
                    "task-image-audio-post-image", "task-audio-image-post-image",
                ],
                "testset": [
                    "task-image-audio-pre-image", "task-audio-image-pre-image",
                    "task-image-audio-post-image", "task-audio-image-post-image",
                ],
                "type": load_type, "permutation": False, "n_channels": n_channels, "n_subjects": n_subjects, "subj_idxs": subj_idxs,
            }),
        ]
    else:
        raise ValueError("ERROR: Unknown dataset {} in train.duin.run_cls.".format(params.train.dataset))
    
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
        #accuracies_validation = []; accuracies_test = []

        # Reset the iteration information of params.
        params.iteration(iteration=0)
        # Initialize model of current time segment.
        model = duin_model(params.model)
        if path_pt_ckpt is not None: model.load_weight(path_pt_ckpt)
        model = model.to(device=params.model.device)
        if params.train.use_graph_mode: model = torch.compile(model)

        # ====== 冻结或检查可训练层 ======
        print("[INFO] Trainable parameters before setting:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.requires_grad}") 

        '''
        for name, param in model.named_parameters():
            if "align_head" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        print("\n[INFO] Trainable parameters after setting:")
        for name, param in model.named_parameters():
            print(f"  {name}: {param.requires_grad}")         
        '''
        # Make an ADAM optimizer for model.
        optim_cfg = utils.DotDict({"name":"adamw","lr":params.train.lr_i,"weight_decay":0.05,})
        optimizer = utils.model.torch.create_optimizer(cfg=optim_cfg, model=model)
        
        train_total_losses = []   # 初始化损失记录列表（每个实验对应一份）
        val_total_losses = []
        test_total_losses = []

        for epoch_idx in range(params.train.n_epochs):   #  训练主循环
            # Update params according to `epoch_idx`, then update optimizer.lr.
            params.iteration(iteration=epoch_idx)
            for param_group_i in optimizer.param_groups: param_group_i["lr"] = params.train.lr_i
            # Record the start time of preparing data.
            time_start = time.time()

            loss_train = utils.DotDict()   # 训练集
            for train_batch in dataset_train:
                # Initialize `batch_i` from `train_batch`.
                batch_i = [
                    train_batch["X"].to(device=params.model.device),
                    train_batch["y"].to(device=params.model.device),
                    train_batch["subj_id"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Train model for current batch.
                y_pred_i, loss_i = _train(batch_i)
                # Numpy the outputs of current batch.
                y_pred_i = y_pred_i.detach().cpu().numpy(); 

                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                #loss_i_total = loss_i.total  # 由 duin_align.forward() 返回
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_train, key_i):
                        loss_train[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_train[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
            for key_i, item_i in loss_train.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_train[key_i] = item_i

            loss_validation = utils.DotDict()   # 验证集
            for validation_batch in dataset_validation:
                # Initialize `batch_i` from `validation_batch`.
                batch_i = [
                    validation_batch["X"].to(device=params.model.device),
                    validation_batch["y"].to(device=params.model.device),
                    validation_batch["subj_id"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Validate model for current batch.
                #y_pred_i, loss_i = _forward(batch_i)
                with torch.no_grad():
                    y_pred_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                y_pred_i = y_pred_i.detach().cpu().numpy()

                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_validation, key_i):
                        loss_validation[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_validation[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
            for key_i, item_i in loss_validation.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_validation[key_i] = item_i

            loss_test = utils.DotDict()   # 测试集
            for test_batch in dataset_test:
                # Initialize `batch_i` from `test_batch`.
                batch_i = [
                    test_batch["X"].to(device=params.model.device),
                    test_batch["y"].to(device=params.model.device),
                    test_batch["subj_id"].to(device=params.model.device),
                ]
                # Get the number of current `batch_i`.
                batch_size_i = len(batch_i[0].detach().cpu().numpy())
                # Test model for current batch.
                #y_pred_i, loss_i = _forward(batch_i)
                with torch.no_grad():
                    y_pred_i, loss_i = _forward(batch_i)
                # Numpy the outputs of current batch.
                y_pred_i = y_pred_i.detach().cpu().numpy()

                for key_i in utils.DotDict.iter_keys(loss_i):
                    utils.DotDict.iter_setattr(loss_i, key_i, utils.DotDict.iter_getattr(loss_i, key_i).detach().cpu().numpy())
                # Record information related to current batch.
                for key_i, item_i in loss_i.items():
                    if hasattr(loss_test, key_i):
                        loss_test[key_i].append(np.array([item_i, batch_size_i], dtype=np.float32))
                    else:
                        loss_test[key_i] = [np.array([item_i, batch_size_i], dtype=np.float32),]
            for key_i, item_i in loss_test.items():
                # Calculate the averaged loss item.
                item_i = np.stack(item_i, axis=0); item_i = np.sum(item_i[:,0] * item_i[:,1]) / np.sum(item_i[:,1])
                # Wrtie loss item back to storation for current epoch.
                loss_test[key_i] = item_i

            ## Write progress to summaries.
            time_stop = time.time()
            msg = (
                "Finish train epoch {:d} in {:.2f} seconds."
            ).format(epoch_idx, time_stop-time_start)
            print(msg); paths.run.logger.summaries.info(msg)

            # Log loss information of train / validation / test
            def log_loss_block(phase_name, loss_dict):
                loss_keys = list(loss_dict.keys())
                #msg = f"Loss({phase_name}): {:.5f} ({})".format(loss_dict[loss_keys[0]], loss_keys[0])
                msg = f"Loss({phase_name}): {loss_dict[loss_keys[0]]:.5f} ({loss_keys[0]})"
                for loss_idx in range(1, len(loss_keys)):
                    msg += "; {:.5f} ({})".format(loss_dict[loss_keys[loss_idx]], loss_keys[loss_idx])
                return msg

            msg = log_loss_block("train", loss_train)
            print(msg); paths.run.logger.summaries.info(msg)
            msg = log_loss_block("validation", loss_validation)
            print(msg); paths.run.logger.summaries.info(msg)
            msg = log_loss_block("test", loss_test)
            print(msg); paths.run.logger.summaries.info(msg)

            train_total_losses.append(loss_train["total"])   # 在每个 epoch 结束时添加记录
            val_total_losses.append(loss_validation["total"])
            test_total_losses.append(loss_test["total"])            

            ## Write progress to tensorboard.
            writer = paths.run.logger.tensorboard
            for phase_name, loss_dict in [("train", loss_train), ("validation", loss_validation), ("test", loss_test)]:
                for key_i, loss_i in loss_dict.items():
                    writer.add_scalar(os.path.join("losses", phase_name, key_i), loss_i, global_step=epoch_idx)
            if epoch_idx == 0:
                msg = summary(model, col_names=("num_params", "params_percent", "trainable",))
                print(msg); paths.run.logger.summaries.info(msg)

            # --- Save checkpoint if validation loss improves ---
            # Track best model based on validation total loss
            if "total" not in loss_validation:
                print("Warning: 'total' not found in loss_validation, skipping checkpoint save for this epoch.")
                continue

            if epoch_idx == 0:
                best_val_loss = loss_validation["total"]
                best_epoch = epoch_idx
                # Save initial model
                save_path = os.path.join(paths.run.ckpt, f"epoch_{epoch_idx:03d}_loss_{best_val_loss:.5f}.pt")
                torch.save(model.state_dict(), save_path)
                msg = f"Initial checkpoint saved at {save_path}"
                print(msg); paths.run.logger.summaries.info(msg)
            else:
                current_val_loss = loss_validation["total"]
                if current_val_loss < best_val_loss:
                    best_val_loss = current_val_loss
                    best_epoch = epoch_idx
                    save_path = os.path.join(paths.run.ckpt, f"best_epoch_{epoch_idx:03d}_loss_{best_val_loss:.5f}.pt")
                    torch.save(model.state_dict(), save_path)
                    msg = f"New best model saved (epoch {epoch_idx}) with val_loss={best_val_loss:.6f}"
                    print(msg); paths.run.logger.summaries.info(msg)

            # --- ✅ 在指定 epoch 保存测试集的 embedding ---
            if (epoch_idx + 1) in [50, 100, 150, 200,250,300]:
                model.eval()
                all_embs = []
                all_labels = []
                with torch.no_grad():
                    for test_batch in dataset_test:
                        X = test_batch["X"].to(params.model.device)
                        y = test_batch["y"].to(params.model.device)
                        subj = test_batch["subj_id"].to(params.model.device)
                        Z_pred, _ = model([X, y, subj])  # (B, 768)

                        emb_pred = Z_pred.detach().cpu().numpy()
                        y_true = y.detach().cpu().numpy()

                        # 用语义 embedding 表找回类别索引
                        emb_table = params.model.emb_table
                        cos_sim = np.dot(y_true, emb_table.T)
                        label_idx = np.argmax(cos_sim, axis=1)

                        combined = np.concatenate([emb_pred, label_idx[:, None]], axis=1)
                        all_embs.append(combined)

                all_embs = np.concatenate(all_embs, axis=0)
                save_path = os.path.join(paths.run.ckpt, f"test_embeddings_epoch_{epoch_idx+1:03d}.npy")
                np.save(save_path, all_embs)
                print(f"[INFO] Saved test embeddings to {save_path}")
                paths.run.logger.summaries.info(f"Saved test embeddings to {save_path}")

        # Log information related to channel weights.
        # ch_weights - (n_subjects, n_channels)
        ch_weights = model.get_weight_i().numpy()
        for subj_idx, subj_cfg_i in enumerate(load_params_i.subjs_cfg):
            # Initialize `ch_names_i` & `ch_weights_i` according to `subj_idx`.
            ch_names_i = subj_cfg_i.ch_names; ch_weights_i = ch_weights[subj_idx,...]
            # Note: Only the former part of `ch_weights_i` corresponds to `ch_names_i`.
            assert len(ch_weights_i.shape) == 1; ch_weights_i = ch_weights_i[:len(ch_names_i)]
            # Get the corresponding channel orders to order channels.
            ch_orders_i = np.argsort(ch_weights_i)[::-1]
            # Log information related to weight distributions of top-k channels.
            top_k = min(10, len(ch_names_i)); msg = (
                "INFO: The top-{:d} channels are {} with weights {}, with channel weight distribution:\n"
            ).format(top_k, [ch_names_i[ch_orders_i[top_idx]] for top_idx in range(top_k)],
                [ch_weights_i[ch_orders_i[top_idx]] for top_idx in range(top_k)])
            msg += log_distr(ch_weights_i)
            print(msg); paths.run.logger.summaries.info(msg)

        # Finish training process of current specified experiment.
        msg = (
            "Finish the (Sementic Embedding Alignment) training process of experiment {}."
        ).format(load_params_i.name)
        print(msg); paths.run.logger.summaries.info(msg)

    writer = paths.run.logger.tensorboard; writer.close()

    # # ====== 绘制并保存 loss 曲线 ======
    # epochs = range(len(train_total_losses))

    # # --- 图1：train vs validation ---
    # plt.figure(figsize=(8, 6))
    # plt.plot(epochs, train_total_losses, label='Train Loss', marker='o')
    # plt.plot(epochs, val_total_losses, label='Validation Loss', marker='s')
    # plt.title('Train vs Validation Total Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # save_path_plot1 = os.path.join(paths.run.ckpt, "loss_train_val(Sem).png")
    # plt.savefig(save_path_plot1, dpi=300)
    # print(f"[INFO] Saved train/val loss plot to {save_path_plot1}")
    # plt.close()

    # # --- 图2：test ---
    # plt.figure(figsize=(8, 6))
    # plt.plot(epochs, test_total_losses, label='Test Loss', color='orange', marker='^')
    # plt.title('Test Total Loss')
    # plt.xlabel('Epoch')
    # plt.ylabel('Loss')
    # plt.legend()
    # plt.grid(True)
    # plt.tight_layout()
    # save_path_plot2 = os.path.join(paths.run.ckpt, "loss_test(Sem).png")
    # plt.savefig(save_path_plot2, dpi=300)
    # print(f"[INFO] Saved test loss plot to {save_path_plot2}")
    # plt.close()

    # After all epochs, log best checkpoint info
    msg = (
        "Best model occurred at epoch {:d} with minimum validation loss {:.6f}."
    ).format(best_epoch, best_val_loss)
    print(msg)
    paths.run.logger.summaries.info(msg)

    # Log the end of current training process.
    msg = "Training finished with dataset {}.".format(params.train.dataset)
    print(msg); paths.run.logger.summaries.info(msg)

# def _forward func
def _forward(inputs):   # 不变
    """
    Forward the model using one-step data. Everything entering this function already be a tensor.

    Args:
        inputs: tuple - The input data, including (X, y_true, subj_id).

    Returns:
        y_pred: (batch_size, n_labels) - The predicted labels.
        loss: DotDict - The loss dictionary.
    """
    global model; model.eval()
    with torch.no_grad(): return model(inputs)

# def _train func
def _train(inputs):   # 不变
    """
    Train the model using one-step data. Everything entering this function already be a tensor.

    Args:
        inputs: tuple - The input data, including (X, y_true, subj_id).

    Returns:
        y_pred: (batch_size, n_labels) - The predicted labels.
        loss: DotDict - The loss dictionary.
    """
    global model, optimizer; model.train()
    # Forward model to get the corresponding loss.
    y_pred, loss = model(inputs)
    # Use optimizer to update parameters.
    optimizer.zero_grad(); loss["total"].backward(); optimizer.step()
    # Return the final `y_pred` & `loss`.
    return y_pred, loss

"""
vis funcs
"""
# def log_distr func
def log_distr(data, n_bins=10, n_hashes=100):   # 不变
    """
    Log information related to data distribution.

    Args:
        data: (n_samples,) - The samples from data distribution.
        n_bins: int - The number of data range, each of which is a base unit to calculate probability.
        n_hashes: int - The total number of hashes (i.e., #) to identify the distribution probability.

    Returns:
        msg: str - The message related to data distribution.
    """
    # Create histogram bins.
    # bins - (n_bins+1,)
    bins = np.linspace(np.min(data), np.max(data), num=(n_bins + 1))
    # Calculate histogram counts.
    # counts - (n_bins,); probs - (n_bins,)
    counts, _ = np.histogram(data, bins=bins); probs = counts / np.sum(counts)
    # Print the histogram.
    msg = "\n"
    for bin_idx in range(len(probs)):
        range_i = "{:.5f} - {:.5f}".format(bins[bin_idx], bins[bin_idx+1]).ljust(20)
        distr_i = "#" * int(np.ceil(probs[bin_idx] * n_hashes))
        msg += "{} | {}\n".format(range_i, distr_i)
    # Return the final `msg`.
    return msg

"""
tool funcs
"""
# def cal_align_loss func
def cal_align_loss(y_pred, y_true):   # 修改
    """
    Calculate the mean squared alignment loss between predicted and target embeddings.
    (Numpy version for logging and aggregation)

    Args:
        y_pred: (*, d_output) - Predicted embedding vectors (after model.forward()).
        y_true: (*, d_output) - Target embedding vectors (ground-truth supervision).

    Returns:
        mse_align: np.float32 - Averaged mean squared error between normalized embeddings.
    """
    # ---- Step 1. L2-normalize both arrays along the last dimension ----
    # Avoid division by zero using eps
    eps = 1e-8
    y_pred_norm = y_pred / (np.linalg.norm(y_pred, axis=-1, keepdims=True) + eps)
    y_true_norm = y_true / (np.linalg.norm(y_true, axis=-1, keepdims=True) + eps)

    # ---- Step 2. Compute elementwise squared difference ----
    diff = y_pred_norm - y_true_norm
    mse_per_sample = np.mean(np.square(diff), axis=-1)  # shape: (batch,)

    # ---- Step 3. Average over batch ----
    mse_align = np.mean(mse_per_sample).astype(np.float32)

    return mse_align

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
    parser = argparse.ArgumentParser("DuIN CLS for brain signals", add_help=False)
    # Add training parmaeters.
    parser.add_argument("--seeds", type=int, nargs="+", default=[42,])
    parser.add_argument("--subjs", type=str, nargs="+", default=["011",])
    parser.add_argument("--subj_idxs", type=int, nargs="+", default=[0,])
    parser.add_argument("--pt_ckpt", type=str, default=None)
    # Return the final `parser`.
    return parser

if __name__ == "__main__":
    import os
    import torch
    # local dep
    from params.duin_params import duin_align_params as duin_params

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

    # Initialize the training process.
    init(duin_params_inst)
    # Loop the training process over random seeds.
    for seed_i in args.seeds:
        # Initialize random seed, then train duin.
        utils.model.torch.set_seeds(seed_i); train()

