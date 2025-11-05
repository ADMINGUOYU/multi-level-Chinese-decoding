# pip install torch torchvision pillow transformers
from typing import Union, List, Literal, Tuple
from PIL import Image
import numpy as np
import torch
from transformers import AutoImageProcessor, ViTModel


def _to_pil(img: Union[str, Image.Image, np.ndarray]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, str):
        return Image.open(img).convert("RGB")
    if isinstance(img, np.ndarray):
        if img.ndim == 2:  # gray
            img = np.stack([img]*3, axis=-1)
        if img.shape[-1] == 4:  # RGBA -> RGB
            img = img[..., :3]
        return Image.fromarray(img.astype(np.uint8)).convert("RGB")
    raise TypeError(f"Unsupported image type: {type(img)}")


@torch.no_grad()
def vit_embed(
    images: Union[str, Image.Image, np.ndarray, List[Union[str, Image.Image, np.ndarray]]],
    model_id: str = "google/vit-base-patch16-224",
    device: Union[str, torch.device] = None,
    output: Literal["cls", "pooler", "patches"] = "cls",
    normalize: bool = True,
) -> Tuple[torch.Tensor, dict]:
    """
    将图片转为 ViT embeddings。

    Args
    ----
    images: 单张或多张图片（路径 / PIL.Image / numpy.ndarray）
    model_id: Hugging Face 上的 ViT 权重
    device: "cuda", "mps", "cpu"；默认自动选择
    output: 
        - "cls": 取 last_hidden_state 的 [CLS] 向量 (B, D)
        - "pooler": 取 model 输出的 pooler_output (B, D)（若无将回退到 CLS）
        - "patches": 取所有 token 序列 (B, N, D)，通常 N=1+num_patches（包含 CLS）
    normalize: 是否对输出做 L2 归一化（常用于相似度检索）

    Returns
    -------
    (embeddings, extra)
        embeddings: torch.Tensor
            - output="cls" → (B, D)
            - output="pooler" → (B, D)
            - output="patches" → (B, N, D)
        extra: dict
            包含以下可能的键：
            - "last_hidden_state": (B, N, D)
            - "attention_mask": (B, N)
            - "patch_grid": (H_p, W_p)  # patch 网格大小（若可推断）
    """
    # 设备
    if device is None:
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
    device = torch.device(device)

    # 准备模型与处理器
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = ViTModel.from_pretrained(model_id).to(device)
    model.eval()

    # 统一成列表
    if not isinstance(images, list):
        images = [images]

    pil_list = [_to_pil(im) for im in images]

    # 预处理
    batch = processor(images=pil_list, return_tensors="pt").to(device)

    # 前向
    outputs = model(**batch)
    last_hidden = outputs.last_hidden_state    # (B, N, D)
    B, N, D = last_hidden.shape

    # 额外信息
    extra = {
        "last_hidden_state": last_hidden,
        "attention_mask": batch.get("attention_mask", None),
    }

    # 尝试估算 patch 网格（去掉 CLS 后还原 HxW）
    # 只有当 (N-1) 是完全平方数时能严格还原
    if N > 1:
        num_patches = N - 1
        side = int((num_patches) ** 0.5)
        if side * side == num_patches:
            extra["patch_grid"] = (side, side)  # (H_p, W_p)

    # 不同输出模式
    if output == "cls":
        emb = last_hidden[:, 0, :]  # [CLS]
    elif output == "pooler":
        pool = getattr(outputs, "pooler_output", None)
        emb = pool if pool is not None else last_hidden[:, 0, :]
    elif output == "patches":
        emb = last_hidden
    else:
        raise ValueError(f"Unknown output={output}")

    # 归一化（仅对向量或序列的最后一维做）
    if normalize:
        if emb.ndim == 2:
            emb = torch.nn.functional.normalize(emb, dim=-1)
        elif emb.ndim == 3:
            emb = torch.nn.functional.normalize(emb, dim=-1)

    return emb, extra


if __name__ == "__main__":
    # 示例：单张图片，取 CLS embedding
    emb, extra = vit_embed("your_image.jpg", output="cls")
    print("CLS embedding shape:", tuple(emb.shape))   # (1, D)
