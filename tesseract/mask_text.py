import cv2
import pytesseract

# Tesseractのパス（必要なら明示的に）
# pytesseract.pytesseract.tesseract_cmd = "/opt/homebrew/bin/tesseract"

# 画像を読み込み
img = cv2.imread("sddefault.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# OCRで文字位置を取得
data = pytesseract.image_to_data(gray, lang='jpn', output_type=pytesseract.Output.DICT)

# 信頼度の高い文字領域にマスクをかける
n = len(data["text"])
for i in range(n):
    if int(data["conf"][i]) > 20 and data["text"][i].strip() != "":
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        img[y:y+h, x:x+w] = cv2.GaussianBlur(img[y:y+h, x:x+w], (15, 15), 0)

# 結果を保存
cv2.imwrite("masked_output.jpg", img)
