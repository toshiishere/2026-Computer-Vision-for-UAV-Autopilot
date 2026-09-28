import cv2
import numpy as np

# Read grayscale image
origin_img = cv2.imread("histogram.jpg")

height, width, channels = origin_img.shape
total_pixels = height * width


def equalize(img):
    # -------------------------
    # 1. Calculate histogram
    # -------------------------
    hist = np.zeros(256, dtype=np.int64)

    for y in range(height):
        for x in range(width):
            intensity = img[y, x]
            hist[intensity] += 1

    # -------------------------
    # 2. Calculate PDF
    # -------------------------
    pdf = hist / total_pixels

    # -------------------------
    # 3. Calculate CDF
    # -------------------------
    cdf = np.zeros(256, dtype=np.float64)

    cdf[0] = pdf[0]

    for i in range(1, 256):
        cdf[i] = cdf[i - 1] + pdf[i]

    # -------------------------
    # 4. Create mapping table
    # -------------------------
    mapping = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        # round(255 * CDF)
        mapping[i] = int(np.floor(255 * cdf[i] + 0.5))

    # -------------------------
    # 5. Apply mapping
    # -------------------------
    output = np.zeros_like(img)

    for y in range(height):
        for x in range(width):
            old_value = img[y, x]
            output[y, x] = mapping[old_value]

    return output


b, g, r = cv2.split(origin_img)
b = equalize(b)
g = equalize(g)
r = equalize(r)
bgr_output = cv2.merge([b, g, r])

hsv_img = cv2.cvtColor(origin_img, cv2.COLOR_BGR2HSV)
h, s, v = cv2.split(hsv_img)
v = equalize(v)
hsv_output = cv2.merge([h, s, v])
hsv_output = cv2.cvtColor(hsv_output, cv2.COLOR_HSV2BGR)



# -------------------------
# Display
# -------------------------
cv2.imshow("Input", origin_img)
cv2.imshow("bgr output", bgr_output)
cv2.imshow("hsv output", hsv_output)

cv2.waitKey(0)
cv2.destroyAllWindows()