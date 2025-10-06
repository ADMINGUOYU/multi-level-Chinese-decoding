import numpy as np

data = np.load(
    "/data0/Users/cchu/Du-IN-main/embeddings/Duin_vit_embeddings.npz",
    allow_pickle=True)

print("Keys:", data.files)

for k in data.files:
    print(k, data[k].shape)

'''
词语图片视觉emb
Keys: ['chars', 'embeddings', 'meta']
chars (61,)
embeddings (61, 768)
meta (6, 2)
'''

print("chars sample:", data["chars"][:10])   # 前10个字符标签
print("embeddings sample:", data["embeddings"][:2, :5])  # 前2条 embedding 的前5维
print("meta:", data["meta"])

'''
chars sample: ['嘴巴' '把' '平静' '豆腐' '面条' '电脑' '头疼' '青菜' '手机' '心情']
embeddings sample: [[-0.00880797  0.01807309 -0.00257216 -0.01295107  0.04373353]
 [-0.02229615  0.0357035   0.0350869  -0.03425682 -0.03498138]]
meta: [['model_id' 'google/vit-base-patch16-224']
 ['encoder' 'vit']
 ['pool' 'mean_patch_tokens']
 ['normalize' 'True']
 ['aggregation' 'per_char']
 ['combine' 'mean']]
'''