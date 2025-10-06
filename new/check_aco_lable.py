import numpy as np

data = np.load(
    "/data0/Users/cchu/Du-IN-main/embeddings/Duin_Acoustic_label.npz",
    allow_pickle=True)

print("Keys:", data.files)

for k in data.files:
    print(k, data[k].shape)

'''
Keys: ['chars', 'embeddings']
chars (61,)
embeddings (61, 2)
'''

print("chars sample:", data["chars"][:61])   
print("embeddings sample:", data["embeddings"][:61])

'''
chars sample: ['嘴巴' '把' '平静' '豆腐' '面条' '电脑' '头疼' '青菜' '手机' '心情' '怎样' '你' '找' '穿' '热水'
 '喝' '碗' '给' '外卖' '预约' '我' '菠萝' '朋友' '漂亮' '米饭' '毛巾' '凳子' '软糖' '厕所' '篮球'
 '丝瓜' '香肠' '拿' '猪肉' '是' '护士' '口渴' '鱼块' '玩' '有' '汤圆' '帮助' '脸盆' '衣服' '放在'
 '关门' '小刀' '醋' '葱花' '钢琴' '蒜泥' '需要' '橙汁' '吃' '家人' '换药' '看' '感觉' '问题' '音乐'
 '愿意']
embeddings sample: [[3 1]
 [3 5]
 [2 4]
 [4 3]
 [4 2]
 [4 3]
 [2 2]
 [1 4]
 [3 1]
 [1 2]
 [3 4]
 [3 5]
 [3 5]
 [1 5]
 [4 3]
 [1 5]
 [3 5]
 [3 5]
 [4 4]
 [4 1]
 [3 5]
 [1 2]
 [2 3]
 [4 4]
 [3 4]
 [2 1]
 [4 4]
 [3 2]
 [4 3]
 [2 2]
 [1 1]
 [1 2]
 [2 5]
 [1 4]
 [4 5]
 [4 4]
 [3 3]
 [2 4]
 [2 5]
 [3 5]
 [1 2]
 [1 4]
 [3 2]
 [1 2]
 [4 4]
 [1 2]
 [3 1]
 [4 5]
 [1 1]
 [1 2]
 [4 2]
 [1 4]
 [2 1]
 [1 5]
 [1 2]
 [4 4]
 [4 5]
 [3 2]
 [4 2]
 [1 4]
 [4 4]]
'''