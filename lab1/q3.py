import cv2
import numpy as np

ddepth = cv2.CV_16S
scale = 1
delta = 0

img = cv2.imread('q3.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1.0)

k1 = np.array([[1, 0, -1],
      [2, 0, -2],
      [1, 0, -1]])

k2 = np.array([[1, 2, 1],
      [0, 0, 0],
      [-1, -2, -1]])

grad_x = cv2.filter2D(blur, ddepth, k1)
grad_y = cv2.filter2D(blur, ddepth, k2)

grad_x_f = grad_x.astype(np.float32)
grad_y_f = grad_y.astype(np.float32)

grad_f = np.sqrt(grad_x_f**2 + grad_y_f**2)
grad = cv2.convertScaleAbs(grad_f)

cv2.imshow('My Image', grad)
cv2.waitKey(0)
cv2.destroyAllWindows