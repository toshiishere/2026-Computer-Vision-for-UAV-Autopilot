import cv2
import numpy as np
import math
import random

origin_img = cv2.imread("q1.png")

height, width, channels = origin_img.shape
total_pixels = height * width

gray = cv2.cvtColor(origin_img, cv2.COLOR_BGR2GRAY)


# -------------------------
# Calculate histogram
# -------------------------
hist = np.zeros(256, dtype=np.int64)

for y in range(height):
    for x in range(width):
        intensity = gray[y, x]
        hist[intensity] += 1


def var(hist, l, r):
    total = 0
    num = 0

    # intensity range: l ~ r-1
    for i in range(l, r):
        total += i * hist[i]
        num += hist[i]

    # Avoid division by zero
    if num == 0:
        return math.inf

    avg = total / num

    variance = 0

    for i in range(l, r):
        variance += hist[i] * (i - avg) ** 2

    # weighted within-class variance
    variance /= total_pixels

    return variance


def calc(thres):
    return var(hist, 0, thres) + var(hist, thres, 256)


best = math.inf
threshold = 0

# 1~255 because threshold=0 creates an empty first group
for i in range(1, 256):
    tmp = calc(i)

    if tmp < best:
        best = tmp
        threshold = i


print("Best threshold:", threshold)
print("Variance:", best)

# Apply threshold to GRAYSCALE image
_, binary_image = cv2.threshold(
    gray,
    threshold,
    255,
    cv2.THRESH_BINARY
)

# cv2.imshow("Binary Image", binary_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# labels = np.zeros(binary_image.shape)
# print("labels.shape", labels.shape)
num_labels, labels = cv2.connectedComponents(binary_image)

colors = []
for i in range(0, num_labels):
    colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))) 

dst = np.zeros((height, width, 3), dtype=np.uint8)
for row in range(height):
    for col in range(width):
        label = labels[row, col]
        if label == 0:
            continue
        dst[row, col] = colors[label]

cv2.imshow("connectedComponents", dst)
cv2.waitKey(0)
cv2.destroyAllWindows()