import cv2
import numpy as np

## 參數調整
VIDEO_PATH = "mot.mp4"
SCALE = 1
BLUR = 5
VAR_THRESHOLD = 50 
LEARNING_RATE = 0.01
OPEN_K = 3
CLOSE_W = 15
CLOSE_H = 61
MIN_AREA_RATIO = 0.004
MAX_WH_RATIO = 3.0


def get_boxes(labels):      # 算每個connected component的框
    boxes = []
    for lab in np.unique(labels):
        if lab == 0:                      
            continue
        ys, xs = np.nonzero(labels == lab)  
        area = len(xs)                      
        boxes.append((area, xs.min(), ys.min(), xs.max(), ys.max()))
    return boxes


def is_valid_box(area, x1, y1, x2, y2, fh, fw):        # 判斷這個區域要不要畫框
    box_w = x2 - x1 + 1
    box_h = y2 - y1 + 1
    T = MIN_AREA_RATIO * fh * fw
    return area > T and box_w <= MAX_WH_RATIO * box_h


cap = cv2.VideoCapture(VIDEO_PATH)
backSub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=VAR_THRESHOLD, detectShadows=True)
shadow_val = backSub.getShadowValue()
open_kernel = np.ones((OPEN_K, OPEN_K), np.uint8)
close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_W, CLOSE_H))

frame_idx = 0
paused = False

while cap.isOpened():
    if not paused:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, None, fx=SCALE, fy=SCALE)
        fh, fw = frame.shape[:2]

        blurred = cv2.GaussianBlur(frame, (BLUR, BLUR), 0) #高斯模糊
        fgmask = backSub.apply(blurred, learningRate=LEARNING_RATE) #用MOG2去背景
        _, nmask = cv2.threshold(fgmask, shadow_val, 255, cv2.THRESH_BINARY) #Threshold去陰影

        nmask = cv2.morphologyEx(nmask, cv2.MORPH_OPEN, open_kernel)
        nmask = cv2.morphologyEx(nmask, cv2.MORPH_CLOSE, close_kernel)

        frame_idx += 1
        num_labels, labels = cv2.connectedComponents(nmask) #找connected component

        for area, x1, y1, x2, y2 in get_boxes(labels):
            if is_valid_box(area, x1, y1, x2, y2, fh, fw):
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2) #畫符合條件的框

        cv2.imshow("frame", frame)
        cv2.imshow("mask", nmask)

    key = cv2.waitKey(33) & 0xFF
    if key == ord(' '):         # 空白鍵暫停
        paused = not paused

    if frame_idx > 0 and (cv2.getWindowProperty("frame", cv2.WND_PROP_VISIBLE) < 1
                          or cv2.getWindowProperty("mask", cv2.WND_PROP_VISIBLE) < 1):
        break

cap.release()
cv2.destroyAllWindows()