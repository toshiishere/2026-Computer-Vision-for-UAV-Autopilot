"""PID ArUco following: face a 15 cm marker at TARGET_DISTANCE centimeters.

1: take off and track; 2: land; 3/Space: hover; T: resume; Q/Esc: exit.
W/S: forward/back; A/D: left/right; Z/X: up/down; C/V: yaw.
Movement keys override PID for 0.5 s after the last key event.
Gains and limits below are starting values and require flight tuning.
"""
from math import atan2, degrees, isfinite, sqrt
import logging
from pathlib import Path
from time import monotonic

from pyimagesearch.pid import PID
from keyboard_djitellopy import MOVEMENT, normalize_key

MARKER_LENGTH = 15.0  # Physical printed marker side length, centimeters.
TARGET_DISTANCE = 100.0  # Camera Z distance, centimeters.
TARGET_ID = 1  # None locks onto the first visible ID until restart.
MAX_SPEED = 50  # RC velocity limit in cm/s.
DEADBAND = 3.0  # Position tolerance in centimeters.
YAW_DEADBAND = 3.0  # Heading tolerance in degrees.
MAX_YAW_SPEED = 50  # Yaw RC limit in degrees/s.
TURN_FIRST_ANGLE = 15.0  # Face the marker before moving forward/back.
MANUAL_TIMEOUT = 0.5  # Seconds after the last movement key event.
STOP = (0, 0, 0, 0)


class RCLogFilter(logging.Filter):
    """Keep high-frequency RC traffic out of the console, including hover."""

    def filter(self, record):
        return not (
            record.levelno == logging.INFO
            and record.getMessage().startswith(
                "Send command (no response expected): 'rc "
            )
        )


class MarkerController:
    def __init__(self):
        self.axes = [PID(kP=1.0, kI=0.1, kD=0.1),
                     PID(kP=1.0, kI=0.07, kD=0.1),
                     PID(kP=2.0, kI=0.2, kD=0.1)]
        self.reset()

    def reset(self):
        for pid in self.axes:
            pid.initialize()
        self.ready = False

    def velocity(self, position):
        x, y, z = map(float, position)
        if not all(isfinite(value) for value in (x, y, z)) or z <= 0:
            self.reset()
            return STOP
        # Turn toward the marker instead of strafing. Positive yaw turns right.
        heading = degrees(atan2(x, z))
        errors = (z - TARGET_DISTANCE, -y, heading)
        if not self.ready:
            for pid, error in zip(self.axes, errors):
                pid.initialize()
                pid.prevError = error  # Avoid derivative kick on acquisition.
            self.ready = True
        speeds = []
        for pid, error, tolerance, limit in zip(
                self.axes, errors, (DEADBAND, DEADBAND, YAW_DEADBAND),
                (MAX_SPEED, MAX_SPEED, MAX_YAW_SPEED)):
            output = pid.update(error, sleep=0)
            speeds.append(0 if abs(error) <= tolerance else
                          int(round(max(-limit, min(limit, output)))))
        forward, vertical, yaw = speeds
        if abs(heading) > TURN_FIRST_ANGLE:
            forward = 0
        return (0, forward, vertical, yaw)


def main():
    import cv2
    from djitellopy import Tello

    calibration = Path(__file__).resolve().with_name('camera.xml')
    fs = cv2.FileStorage(str(calibration), cv2.FILE_STORAGE_READ)
    try:
        if not fs.isOpened():
            raise RuntimeError(f'Cannot open camera calibration: {calibration}')
        intrinsic = fs.getNode('intrinsic').mat()
        distortion = fs.getNode('distortion').mat()
        if intrinsic is None or distortion is None:
            raise RuntimeError('Camera calibration needs intrinsic and distortion matrices')
    finally:
        fs.release()

    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = (aruco.DetectorParameters() if hasattr(aruco, 'DetectorParameters')
                  else aruco.DetectorParameters_create())
    detector = (aruco.ArucoDetector(dictionary, parameters)
                if hasattr(aruco, 'ArucoDetector') else None)
    controller = MarkerController()
    target_id = TARGET_ID
    tracking = False
    previous_frame = None
    last_frame_time = monotonic()
    command = STOP
    manual_command = STOP
    manual_until = 0.0
    window = f'Tello PID - {TARGET_DISTANCE:g} cm'
    drone = Tello()
    rc_log_filter = RCLogFilter()
    drone.LOGGER.addFilter(rc_log_filter)
    try:
        drone.connect()
        print(f'Battery: {drone.get_battery()}%')
        print(__doc__)
        drone.streamon()
        frame_read = drone.get_frame_read()
        cv2.namedWindow(window)
        while True:
            key = normalize_key(cv2.waitKey(30))
            if key in (ord('q'), ord('Q'), 27):
                break
            if key in MOVEMENT:
                if drone.is_flying:
                    manual_command = MOVEMENT[key]
                    manual_until = monotonic() + MANUAL_TIMEOUT
                    controller.reset()
            elif key == ord('1'):
                if not drone.is_flying:
                    drone.takeoff()
                manual_until = 0.0
                tracking = True
                command = STOP
                controller.reset()
            elif key == ord('2'):
                manual_until = 0.0
                tracking = False
                command = STOP
                if drone.is_flying:
                    drone.send_rc_control(*STOP)
                    drone.land()
                controller.reset()
            elif key in (ord('3'), ord(' ')):
                manual_until = 0.0
                tracking = False
                command = STOP
                if drone.is_flying:
                    drone.send_rc_control(*STOP)
                controller.reset()
            elif key in (ord('t'), ord('T')):
                manual_until = 0.0
                command = STOP
                tracking = drone.is_flying
                controller.reset()
            manual_active = drone.is_flying and monotonic() < manual_until
            source = frame_read.frame
            fresh = source is not None and source.size and source is not previous_frame
            if fresh:
                previous_frame = source
                last_frame_time = monotonic()
                # DJITelloPy/PyAV supplies RGB; OpenCV uses BGR.
                frame = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
                command = STOP
                position = None
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(frame)
                else:
                    corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)
                if ids is not None and len(corners):
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    if target_id is None:
                        target_id = int(ids[0, 0])
                    for index, marker_id in enumerate(ids.flatten()):
                        if int(marker_id) != target_id:
                            continue
                        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                            [corners[index]], MARKER_LENGTH, intrinsic, distortion)
                        position = tuple(float(v) for v in tvecs[0].reshape(3))
                        if not all(isfinite(v) for v in position) or position[2] <= 0:
                            position = None
                            break
                        cv2.drawFrameAxes(frame, intrinsic, distortion,
                                          rvecs[0], tvecs[0], MARKER_LENGTH / 2)
                        break
                if tracking and drone.is_flying and position is not None and not manual_active:
                    command = controller.velocity(position)
                else:
                    controller.reset()
                if manual_active:
                    command = manual_command
                status = ('MANUAL' if manual_active else
                          'TRACKING' if tracking and drone.is_flying else 'PAUSED')
                lines = [f'{status} | ID: {target_id} | target Z: {TARGET_DISTANCE:.0f} cm',
                         '1: takeoff/track | 2: land | 3/Space: hover | T: track | Q: quit',
                         'Manual priority: W/S A/D Z/X C/V (0.5 s timeout)']
                if position is None:
                    lines.append('Target missing - manual control' if manual_active else 'Target missing - hover')
                else:
                    x, y, z = position
                    distance = sqrt(x*x + y*y + z*z)
                    lines.append(f'X={x:.1f} Y={y:.1f} Z={z:.1f} cm | distance={distance:.1f} cm')
                if position is not None:
                    lines.append(f'Turn error: {degrees(atan2(position[0], position[2])):.1f} deg')
                lines.append(f'RC: {command}')
                for index, line in enumerate(lines):
                    point = (10, 25 + 25 * index)
                    cv2.putText(frame, line, point, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, line, point, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(window, frame)
            if manual_active:
                command = manual_command
            elif not fresh or not tracking:
                # Never reuse a manual command when its override expires.
                command = STOP
            if not manual_active and monotonic() - last_frame_time > 0.3:
                command = STOP
                controller.reset()
            if drone.is_flying:
                # Continue RC updates while hovering; filter their routine logs.
                drone.send_rc_control(*command)
            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break
    except KeyboardInterrupt:
        pass
    finally:
        try:
            if drone.is_flying:
                try:
                    drone.send_rc_control(*STOP)
                finally:
                    drone.land()
        finally:
            try:
                drone.end()
            finally:
                try:
                    cv2.destroyAllWindows()
                finally:
                    drone.LOGGER.removeFilter(rc_log_filter)


if __name__ == '__main__':
    main()
