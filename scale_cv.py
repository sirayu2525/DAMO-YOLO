import cv2

img = cv2.imread('demo/input2.png')
scale = 4
resized = cv2.resize(img, (img.shape[1]*scale, img.shape[0]*scale), interpolation=cv2.INTER_LANCZOS4)
print(f"size of the picture: {resized.shape[1]} x {resized.shape[0]}")
cv2.imwrite('demo/input_highres_cv2.png', resized)
