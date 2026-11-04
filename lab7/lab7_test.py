from math import atan2, copysign, degrees, isfinite, sqrt
import logging
from pathlib import Path
from time import monotonic

from pyimagesearch.pid import PID

MARKER_LENGTH = 15.0  # Physical printed marker side length, centimeters.
TARGET_DISTANCE = 100.0  # 搜尋階段（看 marker 1 時）維持的 Z 距離，cm。
START_ID = 1  # 看到這個 marker 就開始向右平移。
GOAL_ID = 2  # 平移直到對準這個 marker。
GOAL_X_OFFSET = 0.0  # 對準時 marker 2 的 X（相機座標，cm）。0 = 在畫面正中央。
GOAL_DISTANCE = 115.0  # 對準 marker 2 後要維持的 Z 距離，cm。
HOLD_TIME = 0.5  # 對準並到達 GOAL_DISTANCE 後，維持幾秒。
GOAL_ALIGN_VERTICAL = False  # 對準 marker 2 時是否調整高度。False = 維持目前高度（不上下飛）。
GOAL_Y_OFFSET = 0.0  # 只在 GOAL_ALIGN_VERTICAL=True 時有用：marker 2 的 Y（cm，正=畫面偏下）。0 = 畫面正中央。
SIDE_MOVE_CM = 100  # 維持結束後向左平移的距離，cm（Tello 限制 20~500）。
SEARCH_SPEED = 20  # 向右平移找 marker 2 的速度 (RC 值，約 cm/s)。
SEARCH_TIMEOUT = 15.0  # 超過這麼多秒還沒看到 marker 2 就停下懸停。
MAX_SPEED = 25  # RC velocity limit in cm/s.
MIN_SPEED = 8  # 平移軸最小有效 RC 值：誤差超出 DEADBAND 時至少輸出這麼多（太小 Tello 推不動）。
MIN_YAW_SPEED = 8  # 偏航軸最小有效 RC 值。
DEADBAND = 5.0  # Position tolerance in centimeters.
ALIGN_TOL = 5.0  # 到位判定用的容許誤差 (cm)，比 DEADBAND 寬，避免雜訊讓計時一直歸零。
OUT_GRACE = 0.3  # 到位後短暫掉出範圍的容忍時間（秒），超過才重新計時。
YAW_DEADBAND = 3.0  # Heading tolerance in degrees.
MAX_YAW_SPEED = 30  # Yaw RC limit in degrees/s.
TURN_FIRST_ANGLE = 15.0  # Face the marker before moving forward/back.
MANUAL_TIMEOUT = 0.5  # Seconds after the last movement key event.
STOP = (0, 0, 0, 0)

MOVEMENT = {
    ord('w'): (0, 40, 0, 0), ord('s'): (0, -40, 0, 0),
    ord('a'): (-40, 0, 0, 0), ord('d'): (40, 0, 0, 0),
    ord('z'): (0, 0, 50, 0), ord('x'): (0, 0, -50, 0),
    ord('c'): (0, 0, 0, 30), ord('v'): (0, 0, 0, -30),
}

def normalize_key(key):
    if key < 0:
        return -1
    key &= 0xFF
    return key + 32 if ord('A') <= key <= ord('Z') else key

class RCLogFilter(logging.Filter):
    """Keep high-frequency RC traffic out of the console, including hover."""

    def filter(self, record):
        return not (
            record.levelno == logging.INFO
            and record.getMessage().startswith(
                "Send command (no response expected): 'rc "
            )
        )


def move_left_cm(drone, distance):
    """用 SDK 的 left 指令精準向左飛 distance cm（會阻塞到飛完）。"""
    distance = int(distance)
    drone.send_rc_control(*STOP)
    try:
        # 拉長等待時間，避免逾時後 djitellopy 重送指令造成多飛。
        drone.send_control_command(f'left {distance}', timeout=20)
    except TypeError:
        # 舊版 djitellopy 沒有 timeout 參數。
        drone.move_left(distance)


class MarkerController:
    def __init__(self):
        self.axes = [PID(kP=0.8, kI=0.05, kD=0.05),  # 左右平移
                     PID(kP=0.8, kI=0.05, kD=0.05),  # 前後
                     PID(kP=0.5, kI=0.02, kD=0.05),  # 上下
                     PID(kP=0.7, kI=0.02, kD=0.05)]  # 偏航
        self.reset()

    def reset(self):
        for pid in self.axes:
            pid.initialize()
        self.ready = False

    def velocity(self, position, lateral_target=None, lateral_speed=None,
                 distance=TARGET_DISTANCE, align_vertical=True, vertical_target=0.0):
        """
        lateral_speed  不為 None：左右用固定速度（搜尋模式），前後/上下仍用 PID 維持距離與高度。
        lateral_target 不為 None：左右用 PID 平移到 marker X = lateral_target。
        兩者都是 None：舊行為，轉向對準 marker，不平移。
        distance       前後要維持的 Z 距離 (cm)。
        align_vertical False：不調整高度（上下速度為 0），維持目前高度。
        vertical_target 上下對準時，marker 的 Y 目標值 (cm)。
        """
        x, y, z = map(float, position)
        if not all(isfinite(value) for value in (x, y, z)) or z <= 0:
            self.reset()
            return STOP
        heading = degrees(atan2(x, z))
        if lateral_speed is not None:
            lateral_error, yaw_error = 0.0, 0.0
        elif lateral_target is None:
            lateral_error, yaw_error = 0.0, heading
        else:
            # x 大於目標 -> 向右飛（右移會讓 marker 在畫面中往左移）。
            lateral_error, yaw_error = x - lateral_target, 0.0
        # 左右、深度、上下方向、角度
        vertical_error = -(y - vertical_target) if align_vertical else 0.0
        errors = (lateral_error, z - distance, vertical_error, yaw_error)
        if not self.ready:
            for pid, error in zip(self.axes, errors):
                pid.initialize()
                pid.prevError = error  # Avoid derivative kick on acquisition.
            self.ready = True
        speeds = []
        for pid, error, tolerance, limit, min_speed in zip(
                self.axes, errors,
                (DEADBAND, DEADBAND, DEADBAND, YAW_DEADBAND),
                (MAX_SPEED, MAX_SPEED, MAX_SPEED, MAX_YAW_SPEED),
                (MIN_SPEED, MIN_SPEED, MIN_SPEED, MIN_YAW_SPEED)):
            output = pid.update(error, sleep=0)
            if abs(error) <= tolerance:
                speeds.append(0)
            else:
                value = max(-limit, min(limit, output))
                if abs(value) < min_speed:
                    value = copysign(min_speed, error)  # 四個軸的 RC 方向都和 error 同號
                speeds.append(int(round(value)))
        lateral, forward, vertical, yaw = speeds
        if lateral_speed is not None:
            lateral = lateral_speed
        if lateral_speed is None and lateral_target is None and abs(heading) > TURN_FIRST_ANGLE:
            forward = 0
        return (lateral, forward, vertical, yaw)


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
    tracking = False
    # SEARCH: 向右平移找 marker 2 / ALIGN: 對準 marker 2 並維持距離
    # MOVE_LEFT: 向左平移中 / DONE: 完成後懸停 / TIMEOUT: 找不到 marker 2
    phase = 'SEARCH'
    search_started = monotonic()
    hold_since = None  # 開始「對準且距離到位」的時間；None = 尚未到位。
    out_since = None  # 到位後開始「掉出範圍」的時間；None = 目前沒有掉出。
    last_ctrl_id = None  # 上一次控制器追蹤哪個 marker；換目標時要重置 PID。
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
                    search_started = monotonic()
                    hold_since = None
                    out_since = None
                    last_ctrl_id = None
                    controller.reset()
            elif key == ord('1'):
                if not drone.is_flying:
                    drone.takeoff()
                manual_until = 0.0
                tracking = True
                phase = 'SEARCH'
                search_started = monotonic()
                hold_since = None
                out_since = None
                last_ctrl_id = None
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
                phase = 'SEARCH'
                search_started = monotonic()
                hold_since = None
                out_since = None
                last_ctrl_id = None
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
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(frame)
                else:
                    corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)
                # 同時估計 marker 1 與 marker 2 的位置。tvecs: [X, Y, Z]. rvecs: Rodrigues rotation vector
                positions = {}
                if ids is not None and len(corners):
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    for index, marker_id in enumerate(ids.flatten()):
                        marker_id = int(marker_id)
                        if marker_id not in (START_ID, GOAL_ID):
                            continue
                        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                            [corners[index]], MARKER_LENGTH, intrinsic, distortion)
                        pos = tuple(float(v) for v in tvecs[0].reshape(3))
                        if not all(isfinite(v) for v in pos) or pos[2] <= 0:
                            continue
                        positions[marker_id] = pos
                        cv2.drawFrameAxes(frame, intrinsic, distortion,
                                          rvecs[0], tvecs[0], MARKER_LENGTH / 2)
                start = positions.get(START_ID)
                goal = positions.get(GOAL_ID)
                position = goal if goal is not None else start  # 畫面上顯示用

                if tracking and drone.is_flying and not manual_active:
                    if phase == 'DONE':
                        # 已完成向左平移：懸停。按 1 / T 可重新開始。
                        controller.reset()
                        last_ctrl_id = None
                        command = STOP
                    elif goal is not None:
                        # 看到 marker 2：平移到 X = GOAL_X_OFFSET，並維持 GOAL_DISTANCE。
                        phase = 'ALIGN'
                        if last_ctrl_id != GOAL_ID:
                            controller.reset()
                        last_ctrl_id = GOAL_ID
                        command = controller.velocity(goal, lateral_target=GOAL_X_OFFSET,
                                                      distance=GOAL_DISTANCE,
                                                      align_vertical=GOAL_ALIGN_VERTICAL,
                                                      vertical_target=GOAL_Y_OFFSET)
                        gx, gy, gz = goal
                        # 到位判定用較寬的 ALIGN_TOL；沒開上下對準時，Y 不列入判斷。
                        in_position = (abs(gx - GOAL_X_OFFSET) <= ALIGN_TOL
                                       and abs(gz - GOAL_DISTANCE) <= ALIGN_TOL
                                       and (not GOAL_ALIGN_VERTICAL
                                            or abs(gy - GOAL_Y_OFFSET) <= ALIGN_TOL))
                        if in_position:
                            out_since = None
                            if hold_since is None:
                                hold_since = monotonic()
                            elif monotonic() - hold_since >= HOLD_TIME:
                                phase = 'MOVE_LEFT'
                        elif hold_since is not None:
                            # 容忍短暫掉出範圍；連續超過 OUT_GRACE 秒才重新計時。
                            if out_since is None:
                                out_since = monotonic()
                            elif monotonic() - out_since > OUT_GRACE:
                                hold_since = None
                                out_since = None
                    elif phase == 'SEARCH' and monotonic() - search_started <= SEARCH_TIMEOUT:
                        # 還沒看到 marker 2：持續向右平移。
                        if start is not None:
                            # 看得到 marker 1：向右飛，同時用它維持距離與高度。
                            if last_ctrl_id != START_ID:
                                controller.reset()
                            last_ctrl_id = START_ID
                            command = controller.velocity(start, lateral_speed=SEARCH_SPEED,
                                                          distance=TARGET_DISTANCE)
                        else:
                            # 兩個都看不到：只向右飛（盲飛，受 SEARCH_TIMEOUT 限制）。
                            controller.reset()
                            last_ctrl_id = None
                            command = (SEARCH_SPEED, 0, 0, 0)
                    else:
                        # 找太久，或對準過程中 marker 2 消失：懸停。
                        if phase == 'SEARCH':
                            phase = 'TIMEOUT'
                        hold_since = None
                        out_since = None
                        controller.reset()
                        last_ctrl_id = None
                        command = STOP

                    if phase == 'MOVE_LEFT':
                        # 維持滿 HOLD_TIME：用 SDK 向左平移 SIDE_MOVE_CM（阻塞，視窗會暫停）。
                        print(f'Hold complete. Moving left {SIDE_MOVE_CM} cm...')
                        controller.reset()
                        last_ctrl_id = None
                        hold_since = None
                        out_since = None
                        try:
                            move_left_cm(drone, SIDE_MOVE_CM)
                        except Exception as exc:  # noqa: BLE001
                            print(f'Move left failed: {exc}')
                        phase = 'DONE'
                        command = STOP
                else:
                    controller.reset()
                    last_ctrl_id = None
                if manual_active:
                    command = manual_command
                status = ('MANUAL' if manual_active else
                          f'TRACKING/{phase}' if tracking and drone.is_flying else 'PAUSED')
                seen = [str(i) for i in (START_ID, GOAL_ID) if i in positions]
                shown_z = GOAL_DISTANCE if phase in ('ALIGN', 'MOVE_LEFT', 'DONE') else TARGET_DISTANCE
                lines = [f'{status} | seen: {",".join(seen) or "-"} | target Z: {shown_z:.0f} cm',
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
                if hold_since is not None:
                    lines.append(f'Holding: {monotonic() - hold_since:.1f}/{HOLD_TIME:.1f} s')
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