"用于衡量单个embeddings的分散程度"
import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances

def embedding_dispersion_score(embeddings, n_components=10):
    """
    衡量 embeddings 的分散程度
    输入:
        embeddings: numpy 数组 [num_words, dim]
        n_components: PCA 主成分数
    输出:
        avg_pairwise_dist: 平均两两距离
        pca_explained_ratio: 前 n_components 主成分解释的总方差比
    """
    # 平均成对距离
    dists = pairwise_distances(embeddings, metric="cosine")
    avg_pairwise_dist = dists[np.triu_indices_from(dists, k=1)].mean()

    # PCA 方差解释率
    pca = PCA(n_components=min(n_components, embeddings.shape[1]))
    pca.fit(embeddings)
    pca_explained_ratio = pca.explained_variance_ratio_.sum()

    return avg_pairwise_dist, pca_explained_ratio

# 读取数据
def load_npz_vis(path="Duin_vit_embeddings.npz"):
    data = np.load(path, allow_pickle=True)
    return data["chars"], data["embeddings"], dict(data["meta"])

def load_npz_semantic(path):
    data = np.load(path, allow_pickle=True)
    words = data['words']
    emb_cls = data['emb_cls']
    emb_mean = data['emb_mean']
    emb_max = data['emb_max']
    emb_weighted = data['emb_weighted']
    emb_mixed = data['emb_mixed']
    return words, emb_cls, emb_mean, emb_max, emb_weighted, emb_mixed

words, embeddings, _ = load_npz_vis(\
    "embeddings/Duin_vit_embeddings_vit_per_char.npz")

# words, emb_cls, emb_mean, emb_max, emb_weighted, emb_mixed = load_npz_semantic("embeddings/Duin_bert_embeddings.npz")
# embeddings = emb_weighted  # 选择一种 embedding 方式
# print(embeddings.shape)
# exit()
avg_dist, pca_ratio = embedding_dispersion_score(embeddings)
print(f"Average Pairwise Distance: {avg_dist:.4f}")
print(f"PCA Explained Variance Ratio: {pca_ratio:.4f}")