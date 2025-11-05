
from typing import Sequence, Tuple, Dict, Any, Optional, List
import os
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

try:
    import umap  # type: ignore
    _HAS_UMAP = True
except Exception:
    _HAS_UMAP = False


def _as_numpy(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _normalize_rows(x: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


def _topk_from_scores(scores: np.ndarray, k: int) -> np.ndarray:
    # scores: (N, C), return indices shape (N, k) of top-k (descending)
    return np.argpartition(-scores, kth=min(k, scores.shape[1]-1), axis=1)[:, :k]


def _ranks_from_scores(scores: np.ndarray, true_indices: np.ndarray) -> np.ndarray:
    """
    For each row i, compute the 1-based rank of true_indices[i] under descending scores[i].
    """
    # argsort descending
    order = np.argsort(-scores, axis=1)
    # positions of true class
    # Build inverse index: pos[i, order[i, j]] = j
    inv = np.empty_like(order)
    rows = np.arange(order.shape[0])[:, None]
    inv[rows, order] = np.arange(order.shape[1])[None, :]
    ranks0 = inv[np.arange(order.shape[0]), true_indices]  # 0-based
    return ranks0 + 1  # 1-based


def _label_to_index(labels: Sequence[str]) -> Dict[str, int]:
    return {lab: i for i, lab in enumerate(labels)}


def _indices_for_labels(all_labels: Sequence[str], vocab: Dict[str, int]) -> np.ndarray:
    try:
        return np.array([vocab[l] for l in all_labels], dtype=np.int64)
    except KeyError as e:
        missing = str(e).strip("'")
        raise ValueError(f"Label '{missing}' not found in GT label set.")


def evaluate_semantic_mapping(
    GT_labels_semantic: Sequence[str],
    GT_embeddings_semantic: np.ndarray,
    test_labels_semantic: Sequence[str],
    test_embeddings_semantic: np.ndarray,
    topk: Tuple[int, ...] = (1, 5),
    reducer: str = "pca",  # "pca" | "tsne" | "umap"
    out_dir: str = "semantic_eval_out",
    annotate_prototypes: bool = True,
    random_state: int = 0,
) -> Dict[str, Any]:
    """
    Main entry. See module docstring.
    """
    os.makedirs(out_dir, exist_ok=True)

    # --- Inputs to numpy ---
    GT_labels_semantic = list(GT_labels_semantic)
    test_labels_semantic = list(test_labels_semantic)
    P = _as_numpy(GT_embeddings_semantic)  # (C, D)
    X = _as_numpy(test_embeddings_semantic)  # (N, D)

    C = len(GT_labels_semantic)
    if P.shape[0] != C:
        raise ValueError("GT_embeddings_semantic rows must match number of GT labels.")
    if X.shape[1] != P.shape[1]:
        raise ValueError("Embedding dims mismatch between GT and test.")

    # --- Cosine similarities to class prototypes ---
    Pn = _normalize_rows(P)
    Xn = _normalize_rows(X)
    sims = Xn @ Pn.T  # (N, C)

    # --- Map labels to indices ---
    vocab = _label_to_index(GT_labels_semantic)
    y_true = _indices_for_labels(test_labels_semantic, vocab)  # (N,)

    # --- Predictions via nearest prototype ---
    y_pred = np.argmax(sims, axis=1)

    # --- Overall metrics ---
    overall: Dict[str, Any] = {}
    overall["topk"] = {}
    for k in topk:
        topk_idx = _topk_from_scores(sims, k=k)  # (N, k)
        hits = np.any(topk_idx == y_true[:, None], axis=1).astype(np.float32)
        overall["topk"][f"hits@{k}"] = float(np.mean(hits))

    # MRR
    ranks = _ranks_from_scores(sims, y_true)  # 1-based
    overall["MRR"] = float(np.mean(1.0 / ranks))

    # Top-1 accuracy (same as hits@1)
    overall["accuracy"] = float(accuracy_score(y_true, y_pred))

    # --- Per-class report ---
    # We'll compute per-class accuracy: avg of (pred==true) within each class.
    per_class = []
    for c, lab in enumerate(GT_labels_semantic):
        mask = (y_true == c)
        support = int(mask.sum())
        if support == 0:
            acc_c = np.nan
        else:
            acc_c = float(np.mean(y_pred[mask] == y_true[mask]))
        per_class.append({"label": lab, "index": c, "support": support, "accuracy": acc_c})
    # Save CSV
    import csv
    per_class_csv = os.path.join(out_dir, "per_class_report.csv")
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["label", "index", "support", "accuracy"])
        writer.writeheader()
        writer.writerows(per_class)

    # --- Confusion matrix (C x C) ---
    cm = confusion_matrix(y_true, y_pred, labels=list(range(C)))
    np.save(os.path.join(out_dir, "confusion_matrix.npy"), cm)

    # Plot confusion matrix (no explicit colors; default colormap)
    fig_cm = plt.figure(figsize=(8, 8))
    ax = fig_cm.add_subplot(111)
    im = ax.imshow(cm, interpolation="nearest")
    ax.set_title("Confusion Matrix (labels are indices into GT_labels_semantic)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    # We avoid drawing tick labels for all 61 classes to keep the figure readable by default.
    # Users can adapt to add ticks if needed.
    plt.tight_layout()
    cm_path = os.path.join(out_dir, "fig_confusion.png")
    fig_cm.savefig(cm_path, dpi=160)
    plt.close(fig_cm)

    # --- Clustering-agreement metrics ---
    # Although y_pred is a supervised prototype match, ARI/NMI can still indicate agreement structure.
    ari = float(adjusted_rand_score(y_true, y_pred))
    nmi = float(normalized_mutual_info_score(y_true, y_pred))
    overall["ARI"] = ari
    overall["NMI"] = nmi

    # --- 2D projection for visualization ---
    # Stack [P; X] and reduce jointly so the geometry is shared.
    Z = np.vstack([P, X])  # shape (C+N, D)
    reducer = reducer.lower()
    if reducer == "pca":
        proj = PCA(n_components=2, random_state=random_state).fit_transform(Z)
    elif reducer == "tsne":
        proj = TSNE(n_components=2, random_state=random_state, init="random").fit_transform(Z)
    elif reducer == "umap":
        if not _HAS_UMAP:
            raise ValueError("reducer='umap' requested but umap-learn is not installed.")
        proj = umap.UMAP(n_components=2, random_state=random_state).fit_transform(Z)
    else:
        raise ValueError("reducer must be one of: 'pca', 'tsne', 'umap'")

    proj_P = proj[:C]
    proj_X = proj[C:]

    # Plot: GT prototypes (x) annotated + test points (o)
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111)
    ax.scatter(proj_X[:, 0], proj_X[:, 1], s=8, alpha=0.5, label="test embeddings")
    ax.scatter(proj_P[:, 0], proj_P[:, 1], s=70, marker="x", label="GT prototypes")
    if annotate_prototypes:
        for i, lab in enumerate(GT_labels_semantic):
            ax.annotate(lab, (proj_P[i, 0], proj_P[i, 1]), fontsize=8)
    ax.set_title("2D projection of embeddings (GT prototypes vs test)")
    ax.legend()
    plt.tight_layout()
    proj_path = os.path.join(out_dir, "fig_projection.png")
    fig.savefig(proj_path, dpi=160)
    plt.close(fig)

    # --- Save metrics summary ---
    metrics_path = os.path.join(out_dir, "metrics.json")
    summary = {
        "overall": overall,
        "per_class_csv": per_class_csv,
        "confusion_matrix_png": cm_path,
        "projection_png": proj_path,
        "note": "Confusion matrix axes use class indices; map via GT_labels_semantic."
    }
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    return summary




if __name__ == "__main__":
    import textwrap
    print(textwrap.dedent("""
    This module is not meant to be run directly without data.
    Import `evaluate_semantic_mapping` and call it with your arrays.
    """))