import numpy as np
import faiss

def faiss_topk_cosine(
    GT_embeddings_semantic: np.ndarray,
    test_embeddings_semantic: np.ndarray,
    topk: int = 5,
    use_gpu: bool = False,
    normalize: bool = True,
):
    """
    基于 FAISS 的 Top-K 余弦相似度检索。

    参数
    ----
    GT_embeddings_semantic : np.ndarray
        语义原型库向量，形状 (C, D)
    test_embeddings_semantic : np.ndarray
        测试向量，形状 (N, D)
    topk : int
        取前 K 个最相似的原型
    use_gpu : bool
        是否使用 GPU（需 faiss-gpu）
    normalize : bool
        是否进行 L2 归一化（即使用余弦相似度）
    
    返回
    ----
    scores : np.ndarray
        (N, topk) 对应相似度得分（越大越相似）
    indices : np.ndarray
        (N, topk) 对应的原型索引
    """

    GT = GT_embeddings_semantic.astype(np.float32)
    X = test_embeddings_semantic.astype(np.float32)

    if normalize:
        GT /= np.linalg.norm(GT, axis=1, keepdims=True) + 1e-12
        X  /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

    d = GT.shape[1]  # 特征维度

    # FAISS 的 IndexFlatIP = Inner Product；在归一化后等价于余弦相似度
    index = faiss.IndexFlatIP(d)

    # 可选 GPU 加速
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)

    # 构建索引库
    index.add(GT)

    # 检索
    scores, indices = index.search(X, topk)

    # scores 即 logits（越大越相似）
    return scores, indices



def faiss_topk_labels(
    GT_embeddings_semantic: np.ndarray,
    GT_labels_semantic: list[str],
    target_embeddings: np.ndarray,
    topk: int = 5,
    use_gpu: bool = False,
    normalize: bool = True,
):
    """
    基于 FAISS 的 Top-K 检索：返回每个目标向量的前K个标签和概率(logits)。

    参数
    ----
    GT_embeddings_semantic : np.ndarray
        语义原型库 (C, D)
    GT_labels_semantic : list[str]
        对应的类别标签 (C,)
    target_embeddings : np.ndarray
        需要匹配的目标向量 (N, D)
    topk : int
        取前K个最相似的结果
    use_gpu : bool
        是否使用GPU（若当前faiss版本不支持GPU则自动回退）
    normalize : bool
        是否进行L2归一化（True表示使用余弦相似度）

    返回
    ----
    topk_labels : np.ndarray[str]  (N, K)
        每个样本Top-K预测标签
    topk_probs : np.ndarray[float] (N, K)
        对应的归一化相似度分数（softmax后）
    """

    GT = GT_embeddings_semantic.astype(np.float32)
    X = target_embeddings.astype(np.float32)

    if normalize:
        GT /= np.linalg.norm(GT, axis=1, keepdims=True) + 1e-12
        X  /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12

    d = GT.shape[1]
    index = faiss.IndexFlatIP(d)  # 内积，若归一化则等价于余弦相似度

    # 尝试启用 GPU
    if use_gpu:
        if hasattr(faiss, "StandardGpuResources"):
            try:
                res = faiss.StandardGpuResources()
                index = faiss.index_cpu_to_gpu(res, 0, index)
                print("[FAISS] Using GPU acceleration ✅")
            except Exception as e:
                print(f"[FAISS] GPU init failed, fallback to CPU ({type(e).__name__}: {e})")
        else:
            print("[FAISS] GPU not supported in current FAISS build, using CPU version.")
    else:
        print("[FAISS] Using CPU version.")

    # 添加向量并搜索
    index.add(GT)
    scores, indices = index.search(X, topk)  # (N, K)

    # Softmax 转概率
    exp_scores = np.exp(scores - np.max(scores, axis=1, keepdims=True))
    probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)

    # 匹配标签
    labels = np.array(GT_labels_semantic)
    topk_labels = labels[indices]

    return topk_labels, probs

def compute_topk_accuracy(topk_labels, test_labels_semantic, ks=(1,3,5,10)):
    """
    计算 Top-K 准确率（Hits@K）

    参数
    ----
    topk_labels : np.ndarray[str], shape (N, K)
        每个样本的 Top-K 预测标签
    test_labels_semantic : list[str] or np.ndarray[str], shape (N,)
        每个样本的真实标签
    ks : tuple[int]
        要评估的 K 值集合，例如 (1, 3, 5, 10)

    返回
    ----
    results : dict
        形如 {"hits@1": 0.36, "hits@3": 0.49, ...}
    """

    test_labels = np.asarray(test_labels_semantic)
    N, K = topk_labels.shape
    results = {}

    for k in ks:
        # 取前 k 个预测
        preds_k = topk_labels[:, :k]
        # 判断真实标签是否在前 k 个预测中
        hits = np.any(preds_k == test_labels[:, None], axis=1)
        acc = np.mean(hits)
        results[f"hits@{k}"] = float(acc)

    return results