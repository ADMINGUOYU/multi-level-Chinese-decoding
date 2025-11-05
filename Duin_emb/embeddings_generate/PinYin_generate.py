import numpy as np
import pickle
import pandas as pd
import json
import pypinyin

words = [
    '嘴巴','把','平静','豆腐','面条','电脑','头疼','青菜','手机','心情',
    '怎样','你','找','穿','热水','喝','碗','给','外卖','预约',
    '我','菠萝','朋友','漂亮','米饭','毛巾','凳子','软糖','厕所','篮球',
    '丝瓜','香肠','拿','猪肉','是','护士','口渴','鱼块','玩','有',
    '汤圆','帮助','脸盆','衣服','放在','关门','小刀','醋','葱花','钢琴',
    '蒜泥','需要','橙汁','吃','家人','换药','看','感觉','问题','音乐','愿意'
]
tones = []
for word in words:
    pinyin = pypinyin.lazy_pinyin(word, style=pypinyin.Style.TONE3, errors='replace')
    tone = []
    for i in range(0, len(word)):
        initial = pinyin[i][-1]
        if initial.isdigit():
            tone.append(int(initial))
        else:
            #轻声
            #注：这里正常应该是0，但是由于只有凳子一个词有轻声、为了减少小样本类别，这里直接转化为4
            tone.append(4)
    if len(tone) == 1:
        #占位符
        tone.append(5)
    # print(f"'{word}': {tone},")
    tones.append(np.array(tone))
tones = np.array(tones)
words = np.array(words, dtype=object)
# print(tones)
np.savez(
    f"Duin_Acoustic_label.npz",
    chars=words,
    embeddings=tones
)

