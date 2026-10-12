import cv2
import numpy as np
import math

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

#-----------
#Connect Components
#-----------

# binary: 0 = background, 255 = foreground
height, width = binary_image.shape

# Label image: 0 means background
labels = np.zeros((height, width), dtype=np.int32)
parent = [0]

def make_label():
    label = len(parent)
    parent.append(label)
    return label

def find(x):
    # Find root and compress the path
    if parent[x] != x:
        parent[x] = find(parent[x])
    return parent[x]

def union(a, b):
    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:
        # Merge two equivalent labels
        parent[root_b] = root_a


# Pass 1: Label assignment and equivalence recording
for y in range(height):
    for x in range(width):

        # **Ignore background pixels
        if binary_image[y, x] == 0:
            continue

        # Previous neighbors for 8-connectivity(斜對角也算在相鄰)
        neighbors = []

        if x > 0:
            neighbors.append(labels[y, x - 1])  # Left

        if y > 0:
            if x > 0:
                neighbors.append(labels[y - 1, x - 1])  # Upper-left

            neighbors.append(labels[y - 1, x])  # Up

            if x < width - 1:
                neighbors.append(labels[y - 1, x + 1])  # Upper-right

        # *Ignore background label 0
        neighbors = [n for n in neighbors if n != 0]

        if len(neighbors) == 0:
            # No labeled neighbor: create a new label
            labels[y, x] = make_label()

        else:
            # Assign one of the neighboring labels
            labels[y, x] = min(neighbors)

            # Record equivalence between all neighboring labels
            for n in neighbors:
                union(labels[y, x], n)


# Pass 2: Translate labels to their final roots
for y in range(height):
    for x in range(width):
        if labels[y, x] != 0:
            labels[y, x] = find(labels[y, x])


# Assign color to each connected component
rng = np.random.default_rng(0)  # Seed for reproducibility
num_components = int(labels.max())

colors = rng.integers(
    50, 256,
    size=(num_components + 1, 3), #所有區塊+背景各標上一個顏色
    dtype=np.uint8
)

# Keep background black
colors[0] = [0, 0, 0] 
result = colors[labels]

print("Number of connected components:", num_components)


cv2.imshow("Gray", gray)
cv2.imshow("Binary Image", binary_image)
cv2.imwrite("binary.png", binary_image)

cv2.imshow("connected components", result)
cv2.imwrite("connected_components.png", result)

cv2.waitKey(0)
cv2.destroyAllWindows()