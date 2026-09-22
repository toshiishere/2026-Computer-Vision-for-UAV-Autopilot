import cv2
import numpy as np

img = cv2.imread('q1.jpg')

B, G, R = cv2.split(img)

contrast = 50
brightness = 20

mask = (
    ((B > 1.5 * R) & (B > 1.2 * G)) |
    ((R > 1.5 * B) & (G > 1.5 * B))
)

# Must use float for arithmetic
old_img = img.astype(np.float32)

modified_img = (
    (old_img - 127) * (contrast / 127 + 1)
    + 127
    + brightness
)

# Clamp and convert back to uint8
new_img = np.clip(modified_img, 0, 255).astype(np.uint8)

# Apply brightness/contrast to image
result = img.copy()

# 保留 mask 區域的原始顏色
result[mask] = new_img[mask]

cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("1-2.jpg", result)