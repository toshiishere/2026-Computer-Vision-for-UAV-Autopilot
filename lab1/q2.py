import cv2
import numpy as np

img = cv2.imread('q2.jpg')

h, w, c = img.shape
scale = 3

#用opencvAPI
resized_img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_LINEAR)
cv2.imwrite("output_opencv.jpg", resized_img)

#自行實作
new_h = int(h * scale)
new_w = int(w * scale)

xv, yv = np.meshgrid(np.arange(new_w), np.arange(new_h))

src_x = xv / scale 
src_y = yv / scale 

x1 = np.floor(src_x).astype(int)
y1 = np.floor(src_y).astype(int)
x2 = x1 + 1
y2 = y1 + 1

u = (src_x - x1)[:,:, np.newaxis] #(因為img.shape有hwc 要加一個維度
wy = (src_y - y1)[:,:, np.newaxis]

#避免越界
x1_c = np.clip(x1, 0, w - 1)
x2_c = np.clip(x2, 0, w - 1)
y1_c = np.clip(y1, 0, h - 1)
y2_c = np.clip(y2, 0, h - 1)

Q11 = img[y1_c, x1_c]
Q21 = img[y1_c, x2_c]
Q12 = img[y2_c, x1_c]
Q22 = img[y2_c, x2_c]

value = (1 - u)*(1 - wy)*Q11 + u*(1 - wy)*Q21 + (1 - u)*wy* Q12 + u*wy*Q22
output = np.clip(value, 0, 255).astype(np.uint8)

cv2.imwrite("Bilinear.jpg", output)

cv2.imshow("Original", img)
cv2.imshow("OpenCV", resized_img)
cv2.imshow("Bilinear", output)

cv2.waitKey(0)
cv2.destroyAllWindows()