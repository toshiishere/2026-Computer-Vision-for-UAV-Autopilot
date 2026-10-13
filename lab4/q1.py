import cv2
import numpy as np

cap = cv2.VideoCapture(0)

criteria = cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1

# Vector to store 3D points
obj_points = []

obj3d = np.zeros((54, 3), np.float32)

# Vector to store 2D points
img_points = []

cnt = 0
for i in range(6):
    for j in range(9):
        obj3d[cnt] = (j, i, 0)
        cnt+=1

# print(obj_points)

counter = 0
while counter < 4:

    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('frame', frame)
    # cv2.waitKey(0)

    ret, corner = cv2.findChessboardCorners(gray, (9, 6), None)

    if ret == True:
        corners2 = cv2.cornerSubPix(gray, corner, (11, 11), (-1, -1), criteria)
        img_points.append(corners2)
        obj_points.append(obj3d)
        cv2.drawChessboardCorners(frame, (4, 11), corners2, ret)
        counter += 1

ret, camera_mat, distortion, rotation_vecs, translation_vecs = cv2.calibrateCamera(
    obj_points, img_points, gray.shape[::-1], None, None)

print("Error in projection : \n", ret)
print("\nCamera matrix : \n", camera_mat)
print("\nDistortion coefficients : \n", distortion)
print("\nRotation vector : \n", rotation_vecs)
print("\nTranslation vector : \n", translation_vecs)

f = cv2.FileStorage("file", cv2.FILE_STORAGE_WRITE)
f.write("distortion", distortion)
f.write("intrinsic", camera_mat) 
f.release()