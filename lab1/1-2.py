import cv2
import numpy as np
img = cv2.imread('q1.jpg')

B, G, R = cv2.split(img)

# 找藍點
blue_mask = (
    (B > 100) & (B > 1.5 * R) & (B > 1.2 * G)
)

yellow_mask = ((R > 100) & (G > 100) & (R > 1.5 * B) & (G > 1.5 * B))

contrast=100
brightness=40 

mask = blue_mask | yellow_mask

result = img.copy()

result[mask] = cv2.convertScaleAbs(img[mask], alpha=1.5, beta=20)
# result[mask] = img[mask]

# cv2.imshow("Original", img)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite("1-2.jpg", result)