from PIL import Image, ImageDraw, ImageFont
import matplotlib.font_manager as fm
import os

def chinese_text_to_png(text, output_path="output.png", font_path=None, resolution=(512, 512), bg_color="white", text_color="black"):
    """
    将中文文字渲染成填满画布的 PNG 图像，自动调整字体大小以尽量填满图像。
    
    参数：
        text:         中文字符串
        output_path:  输出PNG文件路径
        font_path:    字体文件路径（.ttf），若为空则尝试自动查找支持中文的字体
        resolution:   图像分辨率 (宽, 高)
        bg_color:     背景颜色
        text_color:   文字颜色
    """
    # 自动查找字体
    if font_path is None:
        font_candidates = [f.fname for f in fm.fontManager.ttflist if "SimHei" in f.name or "MS Gothic" in f.name or "Noto Sans CJK" in f.name]
        if not font_candidates:
            raise RuntimeError("无法找到支持中文的字体，请指定 font_path")
        font_path = font_candidates[0]

    W, H = resolution
    img = Image.new("RGB", (W, H), color=bg_color)
    draw = ImageDraw.Draw(img)

    # 尝试不同字体大小以填满画布
    fontsize = 10
    max_fontsize = 300
    font = None

    for size in range(fontsize, max_fontsize):
        font = ImageFont.truetype(font_path, size)
        bbox = draw.textbbox((0, 0), text, font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        if w > W * 0.95 or h > H * 0.95:
            break

    # 居中绘制
    x = (W - w) // 2 - bbox[0]
    y = (H - h) // 2 - bbox[1]
    img = Image.new("RGB", (W, H), color=bg_color)
    draw = ImageDraw.Draw(img)
    draw.text((x, y), text, font=font, fill=text_color)

    # 去除空白边缘（可选）
    img = img.crop(img.getbbox())

    # 调整回目标尺寸
    img = img.resize((W, H), resample=Image.LANCZOS)
    img.save(output_path)
    print(f"已保存图像至 {output_path}")

if __name__ == "__main__":
    txt = '脑'
    chinese_text_to_png(text=txt, output_path="hanzi.png", resolution=(512, 512))