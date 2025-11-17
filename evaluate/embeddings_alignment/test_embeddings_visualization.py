import plotly.express as px
import numpy as np
from sklearn.decomposition import PCA
from semantic_eval import evaluate_semantic_mapping
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 选择测试的embedding路径
path ='/mnt/afs/250010218/multi-level-Chinese-decoding/\
summaries/2025-11-17/1/train/save_embeddings/semantic_embeddings_epoch_300.npz'

# path='/mnt/afs/250010218/multi-level-Chinese-decoding/summaries/2025-11-09/6/train/save_embeddings/test_embeddings_epoch_300.npy'

# 选择测试类型
test_type = 'Semantic'
# test_type = 'Visual'

log_index = path[58:71]
if log_index[-1] == '/':
    log_index=log_index[:-1]
log_index = log_index.replace('/', '-')



# 指定你的字体文件路径
font_path = '/mnt/afs/250010218/multi-level-Chinese-decoding/Duin_emb/SourceHanSansSC-Normal.otf'

# 添加字体到字体管理器
font_prop = fm.FontProperties(fname=font_path)

# 方法1：设置全局字体
plt.rcParams['font.family'] = font_prop.get_name()
matplotlib.rcParams['axes.unicode_minus'] = False     # 正常显示负号

#注意:
# 测试用 embeddings需要放在 output_embeddings/ 目录下，
# 后续的文件命名没有详细要求，但推荐按照我的命名方式，方便区分不同参数结构
# GT embeddings 可以随意放置，修改路径即可。




# GT embeddings路径
GT_semantic_path = '/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Semantic_GT_bert.npz'
GT_visual_path = '/mnt/afs/250010218/multi-level-Chinese-decoding/GT_embeddings/Duin_Visual_GT_VitPerchar.npz'

#savedir 是 path 去掉output_embeddings/
savedir = path[-29:]

if test_type == 'Semantic':
    savedir = '/mnt/afs/250010218/multi-level-Chinese-decoding/evaluate/embeddings_alignment/semantic_eval_out/'+log_index+'/'+savedir
else:
    savedir ='/mnt/afs/250010218/multi-level-Chinese-decoding/evaluate/embeddings_alignment/visual_eval_out/'+log_index+'/'+savedir

#label list
label_list = ['丝瓜', '你', '关门', '凳子', '厕所', '口渴', '吃',\
               '喝', '嘴巴', '外卖', '头疼', '家人', '小刀', '帮助',\
                  '平静', '心情', '怎样', '感觉', '愿意', '我', '手机',\
                    '找', '把', '护士', '拿', '换药', '放在', '是', '有',\
                          '朋友', '橙汁', '毛巾', '汤圆', '漂亮', '热水', \
                            '猪肉', '玩', '电脑', '看', '碗', '穿', '篮球',\
                                  '米饭', '给', '脸盆', '菠萝', '葱花', '蒜泥',\
                                      '衣服', '豆腐', '软糖', '醋', '钢琴', '问题',\
                                          '需要', '青菜', '面条', '音乐', '预约', '香肠', '鱼块']

#函数定义
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

def load_test_embeddings(path):
    data = np.load(path, allow_pickle=True)
    labels = data[:,-1]
    #把label的元素转化为int
    labels = labels.astype(int)
    labels=[label_list[i] for i in labels]
    labels = np.array(labels)
    embeddings = data[:,0:-1]
    return labels, embeddings

def visualization(emb,label,title,savepath):
    # Plotly 可视化（交互式，不会文字重叠）
    fig = px.scatter(
        x=emb[:, 0],
        y=emb[:, 1],
        text=label,         # 每个点的悬停显示文字
        hover_name=label,   # 鼠标悬停显示
        width=800,
        height=800
    )
    fig.update_traces(
        marker=dict(size=8, opacity=0.7),
        textposition="top center"  # 让点的 label 在上方
    )
    fig.update_layout(
        title=title,
        xaxis_title="PC1",
        yaxis_title="PC2"
    )

    fig.write_html(savepath, auto_open=True)


# 读取GT语义embeddings和模型输出的语义embeddigs
words, emb_cls, emb_mean, emb_max, emb_weighted, emb_mixed = \
    load_npz_semantic(GT_semantic_path)

words_v,emb_v,_ = load_npz_vis(GT_visual_path)
if test_type == 'Semantic':
    GT_labels_semantic = words
    GT_embeddings_semantic = emb_mean
else:
    GT_labels_semantic = words_v
    GT_embeddings_semantic = emb_v

test_labels_semantic, test_embeddings_semantic = \
    load_test_embeddings(path)
print('语义数据格式')
print('GT_labels_semantic:', GT_labels_semantic.shape)
print('GT_embeddings_semantic:', GT_embeddings_semantic.shape)
print('GT_embeddings_semantic Average:', np.mean(GT_embeddings_semantic))
print('test_labels_semantic:', test_labels_semantic.shape)
print('test_embeddings_semantic:', test_embeddings_semantic.shape)
print('test_embeddings_semantic Average:', np.mean(test_embeddings_semantic))

# 归一化
GT_embeddings_semantic = GT_embeddings_semantic / np.linalg.norm(GT_embeddings_semantic, axis=1, keepdims=True)
test_embeddings_semantic = test_embeddings_semantic / np.linalg.norm(test_embeddings_semantic, axis=1, keepdims=True)

results = evaluate_semantic_mapping(
    GT_labels_semantic,         # 长度=61 的汉字列表
    GT_embeddings_semantic,     # 形状 (61, 768)
    test_labels_semantic,       # 长度=329 的标签（取自61类）
    test_embeddings_semantic,   # 形状 (329, 768)
    topk=(1, 3, 5, 10),         #
    reducer="pca",              # 可选 "pca" | "tsne" | "umap"（需安装 umap-learn）
    out_dir=savedir,
    annotate_prototypes=True,   # 是否在图上标注汉字
    random_state=0,
    fig_dpi=300                 # 设置图片DPI (默认160, 推荐300用于出版质量)
)

print(results["overall"])       # 查看总体指标（accuracy、MRR、hits@K、ARI/NMI）
