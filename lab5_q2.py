"""
Lab 5 - Q2: Tello EDU
Stream the Tello camera, detect ArUco markers (15 cm), and overlay
x, y, z, aruco id and the real distance sqrt(x^2 + y^2 + z^2) on the frame.

NOTE: do NOT take off (per the lab instructions). Press ESC or 'q' to quit.

Works with both OpenCV 4.4.0.46 (course version) and OpenCV >= 4.7.
"""
import math
import cv2
import numpy as np
from djitellopy import Tello

MARKER_SIZE = 15  # cm
# djitellopy frames are usually already BGR. If your colors look swapped
# (blue/red inverted), set this to True.
SWAP_RGB_TO_BGR = False

# ---------- 1. Camera calibration (from Q1) ----------
fs = cv2.FileStorage("camera.xml", cv2.FILE_STORAGE_READ)
intrinsic = fs.getNode("intrinsic").mat()
distortion = fs.getNode("distortion").mat()
fs.release()

# ---------- 2. ArUco setup (supports old and new OpenCV API) ----------
NEW_API = hasattr(cv2.aruco, "ArucoDetector")
if NEW_API:  # OpenCV >= 4.7
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters()
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)
    half = MARKER_SIZE / 2
    obj_points = np.array([[-half,  half, 0],
                           [ half,  half, 0],
                           [ half, -half, 0],
                           [-half, -half, 0]], dtype=np.float32)
else:        # OpenCV 4.4.0.46
    dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
    parameters = cv2.aruco.DetectorParameters_create()


def detect(frame):
    if NEW_API:
        corners, ids, _ = detector.detectMarkers(frame)
    else:
        corners, ids, _ = cv2.aruco.detectMarkers(frame, dictionary, parameters=parameters)
    return corners, ids


def estimate_pose(corners):
    """Return list of (rvec, tvec) with tvec in cm, shape (3,)."""
    poses = []
    if NEW_API:
        for c in corners:
            ok, rvec, tvec = cv2.solvePnP(obj_points, c, intrinsic, distortion,
                                          flags=cv2.SOLVEPNP_IPPE_SQUARE)
            poses.append((rvec, tvec) if ok else (None, None))
    else:
        rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
            corners, MARKER_SIZE, intrinsic, distortion)
        for r, t in zip(rvecs, tvecs):
            poses.append((r, t))
    return poses


# ---------- 3. Connect to Tello and start the video stream ----------
drone = Tello()
drone.connect()
print("Battery:", drone.get_battery(), "%")
drone.streamon()
frame_read = drone.get_frame_read()

try:
    while True:
        frame = frame_read.frame
        if frame is None:
            continue
        if SWAP_RGB_TO_BGR:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        frame = frame.copy()

        # ---------- 4. Marker detection ----------
        corners, ids = detect(frame)

        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # ---------- 5. Pose estimation ----------
            poses = estimate_pose(corners)

            for i, (rvec, tvec) in enumerate(poses):
                if rvec is None:
                    continue

                # axes length 10 cm (units must match MARKER_SIZE)
                cv2.drawFrameAxes(frame, intrinsic, distortion, rvec, tvec, 10)

                x, y, z = np.asarray(tvec).reshape(-1)[:3]
                real_distance = math.sqrt(x ** 2 + y ** 2 + z ** 2)

                cx = int(corners[i][0][0][0])
                cy = int(corners[i][0][0][1])

                cv2.putText(frame, f"x: {x:.2f} y: {y:.2f} z: {z:.2f}",
                            (max(cx - 40, 0), max(cy - 25, 15)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                cv2.putText(frame, f"id={ids[i][0]} | dist: {real_distance:.2f} cm",
                            (max(cx - 40, 0), max(cy - 5, 35)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        cv2.imshow("Lab5 Q2 - Tello Marker Distance", frame)
        key = cv2.waitKey(33)
        if key == 27 or key == ord('q'):
            break
finally:
    cv2.destroyAllWindows()
    drone.streamoff()
    drone.end()
