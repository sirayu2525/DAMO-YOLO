from paddleocr import PaddleOCR
from PIL import Image, ImageDraw, ImageFont
import numpy as np

# ① OCRモデルの準備
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# ② 入力画像の読み込み
img_path = "demo/input_highres.png"
image = Image.open(img_path).convert("RGB")
print(f"元画像サイズ: {image.size[0]} x {image.size[1]}")
original_width, original_height = image.size

# ③ 画像をタイル状に分割する関数
def split_image(image, tile_size=512):
    width, height = image.size
    tiles = []
    for y in range(0, height, tile_size):
        for x in range(0, width, tile_size):
            box = (x, y, min(x + tile_size, width), min(y + tile_size, height))
            tile = image.crop(box)
            tiles.append((box, tile))
    return tiles

# ④ 各タイルにOCRを実行し、座標を補正して統合
tile_size = 512
results = []
for (box, tile_img) in split_image(image, tile_size):
    result = ocr.ocr(np.array(tile_img), cls=True)
    if result and result[0]:
        for line in result[0]:
            adjusted_box = [[x + box[0], y + box[1]] for x, y in line[0]]
            text = line[1][0]
            score = line[1][1]
            results.append((adjusted_box, text, score))

# ⑤ 自前で描画（元画像サイズに忠実）
draw = ImageDraw.Draw(image)
font_path = "/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc"
try:
    font = ImageFont.truetype(font_path, size=12)
except IOError:
    font = None  # フォント読み込み失敗時はデフォルトフォント

for box, text, score in results:
    # 矩形を描画（赤）
    draw.line(box + [box[0]], fill="red", width=2)
    # テキストを描画（青）
    if font:
        draw.text(box[0], text, fill="blue", font=font)
    else:
        draw.text(box[0], text, fill="blue")

# 保存
image.save("result_draw_direct_high.png")
print(f"✅ 元画像サイズでのOCR描画が完了しました → 'result_draw_direct.png'")
