#需要测试一下，统一输出格式
import numpy as np
def load_npz_vis(path):
    data = np.load(path, allow_pickle=True)
    return data["chars"], data["embeddings"]

def load_npz_semantic(path,type='mean'):
    data = np.load(path, allow_pickle=True)
    words = data['words']
    emb_cls = data['emb_cls']
    emb_mean = data['emb_mean']
    emb_max = data['emb_max']
    emb_weighted = data['emb_weighted']
    emb_mixed = data['emb_mixed']
    if type == 'mean':
        return words, emb_mean
    else:
        return words, emb_cls, emb_mean, emb_max, emb_weighted, emb_mixed


def load_npz_acoustic(path):
    data = np.load(path, allow_pickle=True)
    return data["chars"], data["embeddings"]



# chars,emb = load_npz_acoustic('GT_embeddings/Duin_Acoustic_label.npz')
# chars,emb = load_npz_semantic('GT_embeddings/Duin_Semantic_GT_bert.npz')
chars,emb = load_npz_vis('GT_embeddings/Duin_Visual_GT_VitPerchar.npz')
print(chars)
print(emb[0].shape)