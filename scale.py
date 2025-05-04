from PIL import Image
import os

# 元画像のパス
input_path = "demo/input2.png"
output_path = "demo/input_highres.png"  # 上書きしたい場合は input_path にする

# 画像読み込み
img = Image.open(input_path)

# 倍率
scale = 6
new_size = (img.width * scale, img.height * scale)

# 高解像度にリサイズ（アンチエイリアスを効かせて滑らかに）
img_highres = img.resize(new_size, resample=Image.LANCZOS)
print(f"size of the picture: {img_highres.size}")

# 保存
img_highres.save(output_path)
