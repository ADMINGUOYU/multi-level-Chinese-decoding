import torch
from transformers import BartTokenizer, BartModel
import numpy as np

model_name = "facebook/bart-base"
#下载模型和分词器
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartModel.from_pretrained(model_name)

# 定义文本
text1 = "今天天气很好，我们去公园散步吧。"
text2 = "人工智能正在改变我们的生活方式。"
text3 = "学习Python编程需要持续的练习和思考。"

# 分词并转换为张量
encoded = tokenizer.encode(text1, return_tensors='pt', add_special_tokens=True)
print(f"Token IDs: {encoded.squeeze().tolist()}")

# 获取模型输出
with torch.no_grad():
    outputs = model(encoded)
    embeddings = outputs.last_hidden_state
    
print(f"Embeddings形状: {embeddings.shape}")
print(f"Embeddings均值: {embeddings.mean().item():.4f}")
print(f"Embeddings标准差: {embeddings.std().item():.4f}")

# 解码回原始文本
decoded_text = tokenizer.decode(encoded.squeeze(), skip_special_tokens=True)
print(f"解码文本: {decoded_text}")