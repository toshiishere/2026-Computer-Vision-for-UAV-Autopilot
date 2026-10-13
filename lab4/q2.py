import cv2
import numpy as np

# ---------- 1. 讀看板圖，滑鼠點四個角 ----------
billboard = cv2.imread('caption.jpg')     
bh, bw = billboard.shape[:2]

pts = []
def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
        pts.append([x, y])

cv2.namedWindow('select')
cv2.setMouseCallback('select', on_mouse)
while len(pts) < 4:
    tmp = billboard.copy()
    for p in pts:
        cv2.circle(tmp, tuple(p), 5, (0, 0, 255), -1)
    cv2.imshow('select', tmp)
    cv2.waitKey(20)
cv2.destroyWindow('select')
# 點擊順序：左上 -> 左下 -> 右上 -> 右下
img_corner = np.float32(pts)

cap = cv2.VideoCapture(0)   
ret, frame = cap.read()
h, w = frame.shape[:2]
cap_corner = np.float32([[0, 0], [0, h - 1], [w - 1, 0], [w - 1, h - 1]])

H = cv2.getPerspectiveTransform(img_corner, cap_corner)

ys, xs = np.mgrid[0:bh, 0:bw]
coords = np.stack([xs.ravel(), ys.ravel(), np.ones(xs.size)]).astype(np.float64)
src = H @ coords
u = src[0] / src[2]          # 除以 w，齊次座標轉回一般座標
v = src[1] / src[2]

# 只保留對應點落在 webcam 畫面內的像素（也就等於看板四邊形內部）
mask = (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
dst_x = xs.ravel()[mask]
dst_y = ys.ravel()[mask]
u = u[mask]
v = v[mask]

# bilinear 的四個鄰點與權重
x0 = np.floor(u).astype(int)
y0 = np.floor(v).astype(int)
x1 = np.minimum(x0 + 1, w - 1)
y1 = np.minimum(y0 + 1, h - 1)
a = (u - x0)[:, None]        # x 方向權重
b = (v - y0)[:, None]        # y 方向權重

while True:
    ret, frame = cap.read()
    if not ret:
        break
    f = frame.astype(np.float32)

    val = ((1 - b) * ((1 - a) * f[y0, x0] + a * f[y0, x1]) +
           b       * ((1 - a) * f[y1, x0] + a * f[y1, x1]))

    out = billboard.copy()
    out[dst_y, dst_x] = val.astype(np.uint8)

    cv2.imshow('result', out)
    if cv2.waitKey(33) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()