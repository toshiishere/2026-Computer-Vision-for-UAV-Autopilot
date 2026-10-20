import math
import cv2
import numpy as np
from djitellopy import Tello

# 1. Load Camera Calibration Parameters
fs = cv2.FileStorage("camera.xml", cv2.FILE_STORAGE_READ)
intrinsic = fs.getNode("intrinsic").mat()
distortion = fs.getNode("distortion").mat()
fs.release()

# 2. Initialize ArUco Dictionary & Parameters
dictionary = cv2.aruco.Dictionary_get(cv2.aruco.DICT_4X4_250)
parameters = cv2.aruco.DetectorParameters_create()

# 3. Initialize Tello Drone Stream
drone = Tello()
drone.connect()
drone.streamon()
frame_read = drone.get_frame_read()

while True:
    frame = frame_read.frame
    if frame is None:
        continue

    # Convert RGB stream from djitellopy to BGR format for OpenCV
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    # 4. Marker Detection
    markerCorners, markerIds, rejectedCandidates = cv2.aruco.detectMarkers(
        frame, dictionary, parameters=parameters
    )

    if markerIds is not None:
        # Draw 2D marker outlines[cite: 6]
        frame = cv2.aruco.drawDetectedMarkers(frame, markerCorners, markerIds)

        # 5. Pose Estimation (Marker size = 15 cm)[cite: 6]
        rvec, tvec, _objPoints = cv2.aruco.estimatePoseSingleMarkers(
            markerCorners, 15, intrinsic, distortion
        )

        for i in range(len(markerIds)):
            # Draw 3D Axes[cite: 6]
            frame = cv2.aruco.drawAxis(
                frame, intrinsic, distortion, rvec[i], tvec[i], 0.1
            )

            # Extract X, Y, Z coordinates (in cm)[cite: 6]
            x = tvec[i][0][0]
            y = tvec[i][0][1]
            z = tvec[i][0][2]

            # Calculate Real Euclidean Distance: sqrt(X^2 + Y^2 + Z^2)[cite: 6]
            real_distance = math.sqrt(x**2 + y**2 + z**2)

            # Anchor text near top-left corner of the detected marker
            corner_x = int(markerCorners[i][0][0][0])
            corner_y = int(markerCorners[i][0][0][1])

            # Overlay X, Y, Z values on frame[cite: 6]
            xyz_str = f"x: {x:.3f} y: {y:.3f} z: {z:.3f}"
            cv2.putText(
                frame, xyz_str, (corner_x - 40, corner_y - 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
            )

            # Overlay ArUco ID and Real Distance[cite: 6]
            info_str = f"id={markerIds[i][0]} | dist: {real_distance:.2f} cm"
            cv2.putText(
                frame, info_str, (corner_x - 40, corner_y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2
            )

    cv2.imshow("Lab 5 - Tello Marker Distance Detection", frame)

    key = cv2.waitKey(33)
    if key == 27 or key == ord('q'):  # Press ESC or 'q' to exit
        break

cv2.destroyAllWindows()
drone.streamoff()