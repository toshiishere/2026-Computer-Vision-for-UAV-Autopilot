from pathlib import Path

from math import sqrt

import cv2
from djitellopy import Tello


# Marker side length and translation vectors are measured in centimeters.
MARKER_LENGTH = 15.0


def main():
    calibration_path = Path(__file__).resolve().with_name("camera.xml")
    fs = cv2.FileStorage(str(calibration_path), cv2.FILE_STORAGE_READ)
    try:
        if not fs.isOpened():
            raise RuntimeError(f"Cannot open camera calibration: {calibration_path}")
        intrinsic = fs.getNode("intrinsic").mat()
        distortion = fs.getNode("distortion").mat()
        if intrinsic is None or distortion is None:
            raise RuntimeError("Camera calibration must contain intrinsic and distortion matrices")
    finally:
        fs.release()

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    if hasattr(aruco, "DetectorParameters"):
        parameters = aruco.DetectorParameters()
    else:
        parameters = aruco.DetectorParameters_create()
    detector = (
        aruco.ArucoDetector(dictionary, parameters)
        if hasattr(aruco, "ArucoDetector")
        else None
    )

    drone = Tello()
    try:
        drone.connect()
        drone.streamon()
        frame_read = drone.get_frame_read()

        while True:
            source_frame = frame_read.frame
            if source_frame is not None and source_frame.size:
                frame = source_frame.copy()
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(frame)
                else:
                    corners, ids, _ = aruco.detectMarkers(
                        frame, dictionary, parameters=parameters
                    )

                if ids is not None and len(corners):
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                        corners, MARKER_LENGTH, intrinsic, distortion
                    )
                    for index, (rvec, tvec) in enumerate(zip(rvecs, tvecs)):
                        cv2.drawFrameAxes(
                            frame, intrinsic, distortion, rvec, tvec,
                            MARKER_LENGTH / 2,
                        )
                        # Camera coordinates: X right, Y down, Z forward.
                        x, y, z = (float(value) for value in tvec.reshape(3))
                        distance = sqrt(x * x + y * y + z * z)
                        label = (
                            f"ID {int(ids[index, 0])}: "
                            f"X={x:.1f} Y={y:.1f} Z={z:.1f} cm  "
                            f"Dist={distance:.1f} cm"
                        )
                        position = (10, 30 + index * 28)
                        # Outline keeps readings visible against bright video.
                        cv2.putText(frame, label, position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 0, 0), 3, cv2.LINE_AA)
                        cv2.putText(frame, label, position,
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                                    (0, 255, 0), 1, cv2.LINE_AA)

                cv2.imshow("drone", frame)

            if (cv2.waitKey(33) & 0xFF) in (ord("q"), 27):
                break
    finally:
        try:
            drone.end()
        finally:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
