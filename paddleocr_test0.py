from paddleocr import PaddleOCR, draw_ocr
from PIL import Image

ocr = PaddleOCR(use_angle_cls=True, lang='japan', use_debug=True)  # 日本語 & 傾き補正あり
img_path = './demo/input2.png'  # 画像のパス
result = ocr.ocr(img_path, cls=True)  # OCR実行

# 結果を画像に描画
image = Image.open(img_path).convert('RGB')
print(f"size of the picture: {image.size}")
boxes = [line[0] for line in result[0]]
txts = [line[1][0] for line in result[0]]
scores = [line[1][1] for line in result[0]]

# 描画して保存
import os
from paddleocr import draw_ocr

# 画像にOCR結果を描画（NumPy配列として返る）
draw_img = draw_ocr(image, boxes, txts, scores, font_path="/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc")
print(f"size of the picture: {draw_img.shape[1]} x {draw_img.shape[0]}")

# NumPy → PIL.Image に変換して保存
PIL_img = Image.fromarray(draw_img)
PIL_img.save('result.png')
print(f"size of the picture: {PIL_img.size}")