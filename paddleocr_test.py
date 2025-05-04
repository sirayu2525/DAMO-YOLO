from paddleocr import PaddleOCR, draw_ocr
from PIL import Image
import numpy as np

# ① OCRモデルの準備
ocr = PaddleOCR(use_angle_cls=True, lang='japan')

# ② 入力画像の読み込み
img_path = "demo/input2.png"
image = Image.open(img_path).convert("RGB")
print(f"元画像サイズ: {image.size[0]} x {image.size[1]}")
# 画像を高解像度にリサイズ（アンチエイリアスを効かせて滑らかに）
# scale = 6
# new_size = (image.width * scale, image.height * scale)
# image = image.resize(new_size, resample=Image.LANCZOS)
# print(f"高解像度画像サイズ: {image.size[0]} x {image.size[1]}")
# 元画像のサイズを保存
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
            # 座標を元画像に補正
            adjusted_box = [[x + box[0], y + box[1]] for x, y in line[0]]
            text = line[1][0]
            score = line[1][1]
            results.append((adjusted_box, text, score))

# ⑤ 描画と保存
boxes = [x[0] for x in results]
txts = [x[1] for x in results]
scores = [x[2] for x in results]

draw_img = draw_ocr(image, boxes, txts, scores, font_path="/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")
Image.fromarray(draw_img).save("result_tiled.png")
print(f"分割OCR結果サイズ: {draw_img.shape[1]} x {draw_img.shape[0]}")

print(f"✅ 分割OCR完了。結果は 'result_tiled.png' に保存されました。")
