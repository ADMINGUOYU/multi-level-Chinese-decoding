import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA

# 读取数据
def load_npz_acoustic(path="Duin_Acoustic_label.npz"):
    data = np.load(path, allow_pickle=True)
    return data["chars"], data["embeddings"]

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

words, embeddings, _ = load_npz_vis("embeddings/Duin_vit_embeddings_vit_per_char.npz")

# words, emb_cls, emb_mean, emb_max, emb_weighted, emb_mixed = load_npz_semantic("embeddings/Duin_bert_embeddings.npz")
# embeddings = emb_max  # 选择一种 embedding 方式

# PCA降到二维
pca = PCA(n_components=2)
emb2d = pca.fit_transform(embeddings)

# Plotly 可视化（交互式，不会文字重叠）
fig = px.scatter(
    x=emb2d[:, 0],
    y=emb2d[:, 1],
    text=words,         # 每个点的悬停显示文字
    hover_name=words,   # 鼠标悬停显示
    width=800,
    height=800
)
fig.update_traces(
    marker=dict(size=8, opacity=0.7),
    textposition="top center"  # 让点的 label 在上方
)
fig.update_layout(
    title="汉字 Embeddings (PCA 2D)",
    xaxis_title="PC1",
    yaxis_title="PC2"
)

fig.write_html('result/Semantic_embedding_max.html', auto_open=True)
