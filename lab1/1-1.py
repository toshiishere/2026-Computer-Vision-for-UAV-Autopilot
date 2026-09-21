import cv2
import numpy as np
img = cv2.imread('q1.jpg')

B, G, R = cv2.split(img)

# 找藍點
blue_mask = (
    (B > 100) & (B > 1.5 * R) & (B > 1.2 * G)
)

# 原圖轉灰階
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 灰階轉回 3 channels
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

# 先讓整張圖變灰階
result = gray_bgr.copy()

# 藍點保留原本顏色
result[blue_mask] = img[blue_mask]

# cv2.imshow("Original", img)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("1-1.jpg", result)