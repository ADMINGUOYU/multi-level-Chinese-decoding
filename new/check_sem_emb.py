import numpy as np

#emb = np.load("/data0/Users/cchu/Du-IN-main/embeddings/emb_epoch0_batch0.npy")

#print("Shape:", emb.shape)

data = np.load("/data0/Users/cchu/Du-IN/embeddings/Duin_bert_embeddings.npz")

print("Keys:", data.files)

for k in data.files:
    print(k, data[k].shape)

'''
单词语义emb
Keys: ['words', 'emb_cls', 'emb_mean', 'emb_max', 'emb_weighted', 'emb_mixed']
words (61,)
emb_cls (61, 768)
emb_mean (61, 768)
emb_max (61, 768)
emb_weighted (61, 768)
emb_mixed (61, 768)
'''

