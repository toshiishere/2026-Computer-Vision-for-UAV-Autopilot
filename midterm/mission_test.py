"""Practice course sections 1, 2, and 3, followed by mission45.

Intended route: marker 1 -> pass right -> marker 2 -> pass left -> marker 3.
Section 2 follows marker 0 until it is absent for one continuous second,
then Section 3 tracks marker 3. Mission45 is enabled only after the marker-3
movement sequence completes, or by pressing 4 to skip ahead manually.

Marker 1: if Z > 125 cm, center and approach; otherwise target X = -50 cm.
Marker 2: if Z > 125 cm, center and approach; otherwise target X = +50 cm.
Marker 3: align position/yaw with its normal at a perpendicular distance of
100 cm and center Y, then run the blocking obstacle-passing sequence.
All velocity_new() branches still use a Z target of 120 cm.

1: take off and track; 2: land; 3/Space: hover; T: resume; Q/Esc: exit.
4: enable mission45 and track marker 4 (already airborne; waits if unseen).
W/S: forward/back; A/D: left/right; Z/X: up/down; C/V: yaw.
Movement keys override PID for 0.5 s after the last key event.
Gains and limits below are starting values and require flight tuning.

After marker 3: once marker 4 is seen, the mission45 routine takes over (own
AlignController PIDs, independent of MarkerController): align marker 4 ->
strafe left until marker 5 -> align with marker 5's normal -> back off to
a perpendicular marker-plane distance of 160 +- 5 cm
-> land immediately.
"""
from math import atan2, degrees, isfinite, sqrt
import logging
from pathlib import Path
from time import monotonic

from pyimagesearch.pid import PID

MARKER_LENGTH = 15.0  # Physical printed marker side length, centimeters.
TARGET_DISTANCE = 120.0  # Camera Z distance, centimeters.
TARGET_DISTANCE_2 = 80
TARGET_DISTANCE_3 = 100.0  # Marker-3 perpendicular plane-distance target (cm).
TARGET_ID = 1  # Display/initial target value; marker filtering below is disabled.
MAX_SPEED = 30  # RC velocity limit in cm/s.
DEADBAND = 3.0  # Position tolerance in centimeters.
YAW_DEADBAND = 3.0  # Heading tolerance in degrees.
MAX_YAW_SPEED = 60  # Yaw RC limit in degrees/s.
TURN_FIRST_ANGLE = 15.0  # Face the marker before moving forward/back.
MANUAL_TIMEOUT = 0.5  # Seconds after the last movement key event.
STOP = (0, 0, 0, 0)  # RC order: left/right, forward/back, up/down, yaw.

# ===== mission45：看到 marker 4 之後的流程 (參數與 PID 都獨立於上面 stage13 的設定) =====
ID_START = 4             # 看到它就接手：先對準 marker 4
ID_END = 5               # 對準 marker 4 後向左平移要找的 marker
M45_TARGET_DISTANCE = 150.0  # Perpendicular marker-plane alignment distance (cm).
M45_DEADBAND = 3.0       # mission45 位置容許誤差 (cm)，與 stage13 的 DEADBAND 無關
M45_MAX_SPEED = 30       # mission45 PID 輸出的 RC 速度上限
STRAFE_SPEED = 30        # 向左平移速度 (送出 -STRAFE_SPEED；負 = 向左)
STRAFE_TIMEOUT = 20.0    # 平移超過這個秒數還沒看到 marker 5 就放棄
ALIGN_HOLD_FRAMES = 3    # 需「連續」N 幀都在誤差內，才算真的對準
BACK_DISTANCE = 150.0    # Perpendicular marker-plane back-off distance (cm).
BACK_TOL = 5.0           # Land immediately within this plane-distance tolerance.
BACK_TIMEOUT = 60.0      # 後退超過這個秒數還沒到位就直接降落
ALIGN_4, STRAFE_LEFT, ALIGN_5, BACK_OFF, DONE = (
    'ALIGN_4', 'STRAFE_LEFT', 'ALIGN_5', 'BACK_OFF', 'DONE')
M45_YAW_DEADBAND = 3.0  # Normal heading tolerance, degrees.
M45_MAX_YAW_SPEED = 30
ALIGN_YAW_KP = 0.4  # IDs 3/4/5: half the original MarkerController yaw P (0.8).
# M45_TARGET_DISTANCE and BACK_DISTANCE are perpendicular plane distances (cm).

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


class MarkerController:
    """Convert camera-relative marker errors into bounded RC commands."""

    def __init__(self):
        # velocity_new() axis order: lateral X, depth Z, vertical Y, heading.
        self.axes = [PID(kP=1.0, kI=0.1, kD=0.1),
                     PID(kP=1.0, kI=0.1, kD=0.1),
                     PID(kP=1.0, kI=0.1, kD=0.1),
                     PID(kP=0.8, kI=0.02, kD=0.1)]
        self.reset()

    def reset(self):
        # Clear accumulated integral and derivative history after interruptions.
        for pid in self.axes:
            pid.initialize()
        self.ready = False

    def set_follow(self):
        # Clear accumulated integral and derivative history after interruptions.
        self.axes = [PID(kP=0, kI=0, kD=0),
                    PID(kP=1.0, kI=0.1, kD=0.1),
                    PID(kP=1, kI=0.07, kD=0.1),
                    PID(kP=2.0, kI=0.3, kD=0.1)]
    def unset_set_follow(self):
        # Clear accumulated integral and derivative history after interruptions.
        self.axes = [PID(kP=1.0, kI=0.1, kD=0.1),
                     PID(kP=1.0, kI=0.1, kD=0.1),
                     PID(kP=1.0, kI=0.1, kD=0.1),
                     PID(kP=0.8, kI=0.02, kD=0.1)]
    def velocity(self, position):
        """Older forward/vertical/yaw follower; main() uses velocity_new()."""
        x, y, z = map(float, position)
        if not all(isfinite(value) for value in (x, y, z)) or z <= 0:
            self.reset()
            return STOP
        # Turn toward the marker instead of strafing. Positive yaw turns right.
        heading = degrees(atan2(x, z))
        errors = (z - TARGET_DISTANCE, -y * 0, heading) # 深度、上下方向、角度
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


    def velocity_new(self, position, target_x, stage = -1, target_y = 0, target_z = TARGET_DISTANCE):
        """Strafe toward target_x while regulating Z to TARGET_DISTANCE.

        stage=1 disables vertical correction; any other stage enables it.
        This argument is a controller mode, not a stored mission phase.
        Camera X points right, Y points down, and Z points forward (cm).
        The current vertical error is +(Y-target_y), as implemented below.
        Heading PID is calculated, but the returned yaw command is always zero.
        """
        x, y, z = map(float, position)
        mult = 0 if stage == 1 else 1 # 預設考慮y方向error
        mult_yaw = 1 if stage == 2 else 0 # 預設不考慮轉頭error
        if not all(isfinite(value) for value in (x, y, z)) or z <= 0:
            self.reset()
            return STOP
        heading = degrees(atan2(x, z))
        errors = (x-target_x, z - target_z, (target_y - y) * mult, heading)
        if not self.ready:
            for pid, error in zip(self.axes, errors):
                pid.initialize()
                pid.prevError = error  # Avoid derivative kick on acquisition.
            self.ready = True
        speeds = []
        for pid, error, tolerance, limit in zip(
                self.axes, errors, (DEADBAND, DEADBAND, DEADBAND, YAW_DEADBAND),
                (MAX_SPEED, MAX_SPEED, MAX_SPEED, MAX_YAW_SPEED)):
            output = pid.update(error, sleep=0)
            speeds.append(0 if abs(error) <= tolerance else
                          int(round(max(-limit, min(limit, output)))))
        parallel, forward, vertical, yaw = speeds

        # if abs(heading) > TURN_FIRST_ANGLE:
        #     forward = 0
        return (parallel, forward, vertical, yaw * mult_yaw)  # Keep the course heading fixed.



class AlignController:
    """Align position and yaw with the marker-plane normal for IDs 3, 4, 5.

    與 MarkerController 完全分開 (各自的 PID 物件、增益、容許誤差、限幅)，
    stage13 的 set_follow()/unset_set_follow() 不會影響它。
    """

    def __init__(self):
        self.axes = [PID(kP=0.7, kI=0.002, kD=0.1),     # 左右
                     PID(kP=0.7, kI=0.002, kD=0.15),   # 前後
                     PID(kP=ALIGN_YAW_KP, kI=0.02, kD=0.1),  # independent yaw
                     PID(kP=1.0, kI=0.1, kD=0.1)]      # optional vertical
        self.reset()

    def reset(self):
        # 清除積分/微分記憶，下次重新取得第一筆誤差
        for pid in self.axes:
            pid.initialize()
        self.ready = False

    def velocity(self, position, normal, target_distance=M45_TARGET_DISTANCE,
                 target_y=None):
        """Return (RC, aligned), using the plane normal in camera coordinates.

        The normal points into the marker (positive camera Z). At the current
        height, target a point on its normal in X/Z and face the marker plane.
        target_y enables vertical centering for marker 3; otherwise vertical RC
        remains zero. Pitch/roll are not controlled.
        """
        x, y, z = map(float, position)
        nx, ny, nz = map(float, normal)
        length = sqrt(nx*nx + ny*ny + nz*nz)
        if (not all(isfinite(v) for v in (x, y, z, nx, ny, nz, length,
                                          target_distance))
                or z <= 0 or length <= 0 or target_distance <= 0
                or (target_y is not None and not isfinite(target_y))):
            self.reset()
            return STOP, False

        nx, ny, nz = nx / length, ny / length, nz / length
        if nz < 0:
            nx, ny, nz = -nx, -ny, -nz
        horizontal_squared = nx*nx + nz*nz
        if nz <= 0 or horizontal_squared < 0.01:
            self.reset()
            return STOP, False
        # n dot t is perpendicular distance. Account for fixed camera height
        # when solving for target X/Z, including markers tilted vertically.
        scale = (target_distance - ny*y) / horizontal_squared
        heading = degrees(atan2(nx, nz))
        errors = (x - scale*nx, z - scale*nz, heading)
        if target_y is not None:
            errors += (target_y - y,)

        if not self.ready:
            # 第一次取得誤差時，把 prevError 設成目前誤差，避免微分項暴衝
            for pid, error in zip(self.axes, errors):
                pid.initialize()
                pid.prevError = error
            self.ready = True

        speeds = []
        tolerances = (M45_DEADBAND, M45_DEADBAND, M45_YAW_DEADBAND)
        limits = (M45_MAX_SPEED, M45_MAX_SPEED, M45_MAX_YAW_SPEED)
        if target_y is not None:
            tolerances += (DEADBAND,)
            limits += (MAX_SPEED,)
        for pid, error, tolerance, limit in zip(self.axes, errors, tolerances, limits):
            output = pid.update(error, sleep=0)
            speeds.append(0 if abs(error) <= tolerance else
                          int(round(max(-limit, min(limit, output)))))

        in_tolerance = all(abs(e) <= tol for e, tol in zip(errors, tolerances))
        lr, fb, yaw = speeds[:3]
        vertical = speeds[3] if target_y is not None else 0
        return (lr, fb, vertical, yaw), in_tolerance


def main():
    import cv2
    from djitellopy import Tello

    stage3_flag = False
    stage3_hold = 0
    section = 1
    marker0_missing_since = None
    stage4_start_flag = False
    mission45_enabled = False
    m45_state = None           # None = 還沒看到 marker 4，仍跑 stage13 原本邏輯
    hold = 0                   # mission45 已連續對準的幀數
    strafe_start = 0.0         # 開始向左平移的時間
    back_start = 0.0           # 開始後退的時間
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

    # Support both newer and older OpenCV ArUco detector APIs.
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = (aruco.DetectorParameters() if hasattr(aruco, 'DetectorParameters')
                  else aruco.DetectorParameters_create())
    detector = (aruco.ArucoDetector(dictionary, parameters)
                if hasattr(aruco, 'ArucoDetector') else None)
    controller = MarkerController()
    align3 = AlignController()  # Independent PID history for marker 3.
    align = AlignController()  # mission45 專用，PID 與 controller 完全獨立
    target_id = TARGET_ID
    tracking = False
    previous_frame = None
    last_frame_time = monotonic()
    command = STOP
    manual_command = STOP
    stage4_dis = 0
    manual_until = 0.0
    window = f'Tello PID - {TARGET_DISTANCE:g} cm'
    drone = Tello()
    rc_log_filter = RCLogFilter()
    drone.LOGGER.addFilter(rc_log_filter)
    try:
        drone.connect()
        print(f'Battery: {drone.get_battery()}%')
        print(__doc__)
        # sleep()
        drone.streamon()
        frame_read = drone.get_frame_read()
        cv2.namedWindow(window)
        while True:
            # Keyboard input: manual commands temporarily override automatic tracking.
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
                m45_state, hold = None, 0   # 重新從 stage13 開頭開始
                mission45_enabled = False
                stage3_flag = False
                stage4_start_flag = False
                section = 1
                marker0_missing_since = None
                stage3_hold = 0
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
            elif key == ord('4'):
                mission45_enabled = True
                manual_until = 0.0
                command = STOP
                tracking = drone.is_flying
                stage3_flag = False
                stage4_start_flag = False
                controller.reset()
                align3.reset()
                print('mission45 enabled manually: waiting for marker 4')
            elif key in (ord('t'), ord('T')):
                manual_until = 0.0
                command = STOP
                tracking = drone.is_flying
                controller.reset()
            manual_active = drone.is_flying and monotonic() < manual_until
            if manual_active or not tracking:
                stage3_hold = 0
                marker0_missing_since = None
                # 手動或暫停時清掉 mission45 的 PID 記憶與對準計數，交還控制時重新開始
                align.reset()
                hold = 0
            if manual_active and m45_state == STRAFE_LEFT:
                strafe_start = monotonic()   # 手動期間不計入平移超時
            if manual_active and m45_state == BACK_OFF:
                back_start = monotonic()     # 手動期間不計入後退超時
            source = frame_read.frame
            # Only run vision/PID when the video reader supplies a new frame object.
            fresh = source is not None and source.size and source is not previous_frame
            if fresh:
                previous_frame = source
                last_frame_time = monotonic()
                # DJITelloPy/PyAV supplies RGB; OpenCV uses BGR.
                frame = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
                command = STOP
                position = None
                positions = {}  # {marker id: (x, y, z)}，mission45 用
                normals = {}  # Unit marker-plane normals in camera coordinates.
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(frame)
                else:
                    corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)
                if ids is not None and len(corners):
                    order = ids.flatten().argsort()
                    ids = ids[order]
                    corners = [corners[index] for index in order]
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    if target_id is None:
                        target_id = int(ids[0, 0])
                    for index, marker_id in enumerate(ids.flatten()):
                        # Filtering is disabled: every detected marker is processed.
                        # position ends up belonging to the last valid marker, while
                        # the control branches below use ids[0, 0] (the first ID).
                        # if int(marker_id) != target_id:
                        #     continue
                        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers( # tvecs: [X, Y, Z]. rvecs: Rodrigues rotation vector
                            [corners[index]], MARKER_LENGTH, intrinsic, distortion)
                        position = tuple(float(v) for v in tvecs[0].reshape(3))
                        if not all(isfinite(v) for v in position) or position[2] <= 0:
                            position = None
                            break
                        cv2.drawFrameAxes(frame, intrinsic, distortion,
                                          rvecs[0], tvecs[0], MARKER_LENGTH / 2)
                        positions[int(marker_id)] = position
                        if int(marker_id) in (3, ID_START, ID_END):
                            rotation, _ = cv2.Rodrigues(rvecs[0])
                            normal = rotation[:, 2]  # Marker-local Z in camera frame.
                            if normal[2] < 0:
                                normal = -normal
                            normals[int(marker_id)] = tuple(float(v) for v in normal)
                        # break
                # Only advance from Section 2 after a full second without ID 0.
                if (tracking and drone.is_flying and not manual_active
                        and m45_state is None and not mission45_enabled):
                    marker0_visible = ids is not None and 0 in ids.flatten()
                    if marker0_visible and section != 2:
                        section = 2
                        marker0_missing_since = None
                        stage3_flag = False
                        stage4_start_flag = False
                        controller.set_follow()
                        controller.reset()
                    if section == 2:
                        if marker0_visible:
                            marker0_missing_since = None
                        elif marker0_missing_since is None:
                            marker0_missing_since = monotonic()
                        elif monotonic() - marker0_missing_since >= 1.0:
                            section = 3
                            marker0_missing_since = None
                            controller.unset_set_follow()
                            controller.reset()
                            print('ID 0 missing for 1 second -> Section 3')
                    if section in (2, 3):
                        position = positions.get(0 if section == 2 else 3)
                # ---- 看到 marker 4：停止 stage13 流程，改由 mission45 接手 ----
                if (tracking and drone.is_flying and not manual_active
                        and mission45_enabled and m45_state is None and ID_START in positions):
                    m45_state, hold = ALIGN_4, 0
                    stage4_start_flag = False   # 不再繼續上升
                    align.reset()
                    print('看到 marker 4 -> 進入 mission45 流程')

                if not (tracking and drone.is_flying and not manual_active
                        and m45_state is None and not mission45_enabled
                        and section == 3 and 3 in positions
                        and not stage4_start_flag):
                    stage3_hold = 0
                    align3.reset()

                if tracking and drone.is_flying and not manual_active and m45_state is not None:
                    # ================= mission45 狀態機 (只用 align 的獨立 PID) =================
                    command = STOP
                    if m45_state == ALIGN_4:
                        # Align with marker 4's normal at the plane-distance target.
                        if ID_START in positions:
                            command, ok = align.velocity(positions[ID_START], normals[ID_START])
                            hold = hold + 1 if ok else 0   # 連續計數，一幀不合格就歸零
                            if hold >= ALIGN_HOLD_FRAMES:
                                m45_state = STRAFE_LEFT
                                strafe_start = monotonic()
                                align.reset()
                                command = STOP
                                print('已對準 marker 4 -> 開始向左平移')
                        else:
                            align.reset()   # 看不到 marker 4：懸停等待
                            hold = 0

                    elif m45_state == STRAFE_LEFT:
                        # 固定速度向左平移，直到看到 marker 5
                        if ID_END in positions:
                            m45_state = ALIGN_5     # 先煞車，下一幀開始置中
                            hold = 0
                            align.reset()
                            command = STOP
                            print('看到 marker 5 -> 開始置中')
                        elif monotonic() - strafe_start > STRAFE_TIMEOUT:
                            m45_state = DONE        # 超時保護
                            command = STOP
                            print('平移超時，沒看到 marker 5 -> 懸停')
                        else:
                            command = (-STRAFE_SPEED, 0, 0, 0)

                    elif m45_state == ALIGN_5:
                        # Align with marker 5's normal at the plane-distance target.
                        if ID_END in positions:
                            command, ok = align.velocity(positions[ID_END], normals[ID_END])
                            hold = hold + 1 if ok else 0
                            if hold >= ALIGN_HOLD_FRAMES:
                                m45_state = BACK_OFF
                                back_start = monotonic()
                                hold = 0
                                align.reset()
                                command = STOP
                                print(f'marker 5 已置中 -> 後退到相距 {BACK_DISTANCE:.0f} cm')
                        else:
                            align.reset()   # 暫時看不到 marker 5：原地懸停等它回來
                            hold = 0

                    elif m45_state == BACK_OFF:
                        # Back off along the normal; land at the plane-distance target.
                        land_now = False
                        if ID_END in positions:
                            command, _ = align.velocity(positions[ID_END], normals[ID_END],
                                                         target_distance=BACK_DISTANCE)
                            distance5 = sum(p*n for p, n in zip(positions[ID_END], normals[ID_END]))
                            if abs(distance5 - BACK_DISTANCE) <= BACK_TOL:
                                print(f'Plane distance={distance5:.1f} cm -> land')
                                land_now = True
                        else:
                            align.reset()
                            hold = 0
                            command = STOP
                        if not land_now and monotonic() - back_start > BACK_TIMEOUT:
                            print('後退超時 -> 直接降落')
                            land_now = True
                        if land_now:
                            command = STOP
                            drone.send_rc_control(*STOP)   # 先停止動作再降落
                            drone.land()
                            tracking = False
                            m45_state = DONE

                    else:  # DONE：懸停。按 1 可從頭重新開始。
                        command = STOP

                elif (tracking and drone.is_flying and not manual_active
                      and ((not mission45_enabled and position is not None)
                           or stage4_start_flag)):
                    # SECTION 1, first obstacle: marker 1.
                    # The test is one-sided Z error, not full X/Y/Z alignment.
                    if section == 1 and ids is not None and int(ids[0, 0]) == 1:
                        print("ids = 1")
                        X = position[0]
                        Y = position[1]
                        Z = position[2]
                        # position_new = tuple([X, Y, Z])

                        print(f"Z-TARGET_DISTANCE = {Z-TARGET_DISTANCE}")
                        if Z-TARGET_DISTANCE > 5:
                            # Z > 125 cm: approach at centered X; hold vertical RC at 0.
                            command = controller.velocity_new(position, 0, 1)
                        else:
                            # Target marker X=-50: strafe right so it appears to the left.
                            command = controller.velocity_new(position, -50, 1, 0)
                            
                    # SECTION 1, second obstacle: marker 2.
                    elif section == 1 and ids is not None and int(ids[0, 0]) == 2:
                        print("ids = 2")
                        X = position[0]
                        Y = position[1]
                        Z = position[2]
                        # position_new = tuple([X, Y, Z])

                        # command = controller.velocity(position)
                        print(f"Z-TARGET_DISTANCE = {Z-TARGET_DISTANCE}")
                        if Z-TARGET_DISTANCE > 10:
                            # Z > 125 cm: approach centered, without vertical correction.
                            command = controller.velocity_new(position, 0, 1)
                        else:
                            # Target marker X=+50: strafe left so it appears to the right.
                            # command = controller.velocity_new(position, 50, 1)
                            print("MOVE LEFT")
                            drone.move_left(100)

                    # SECTION 2
                    elif section == 2 and 0 in positions:
                        X = position[0]
                        Y = position[1]
                        Z = position[2]
                        print("ids = 0")
                        print(f"Z-TARGET_DISTANCE = {Z-TARGET_DISTANCE_2}")
                        command = controller.velocity_new(position, 0, 2, 0, TARGET_DISTANCE_2)

                    # SECTION 3: marker 3 after the marker-0 loss timeout.
                    elif section == 3 and 3 in positions and not stage4_start_flag:
                        print("id = 3")
                        command, aligned = align3.velocity(
                            positions[3], normals[3],
                            target_distance=TARGET_DISTANCE_3, target_y=0)
                        stage3_hold = stage3_hold + 1 if aligned else 0
                        if stage3_hold >= ALIGN_HOLD_FRAMES:
                            stage3_hold = 0
                            print("Goes DOWNNN")
                            # command = controller.velocity_new(position=position, target_x=0, stage=3, target_y=-60, target_z=TARGET_DISTANCE_3)
                            stage3_flag = True
                    elif ids is not None and int(ids[0, 0]) == 4:
                        stage4_start_flag = False

                    if stage3_flag == True:
                        # Blocking SDK command: fly forward 150 cm before the loop resumes.
                        # The flag resets here, but marker 3 can set it again next frame.
                        # The RC command above is sent only later, after this call returns.
                        stage3_flag = False
                        print("drone.move_forward(300)")
                        drone.move_down(60)
                        drone.move_forward(270)
                        drone.rotate_clockwise(90)
                        drone.move_back(80)
                        mission45_enabled = True  # All four blocking moves succeeded.
                        stage4_start_flag = True
                        stage4_dis = 50

                    if stage4_start_flag:
                        # stage4_start_flag = False
                        print("MOVE UP")
                        drone.move_up(max(20, int(stage4_dis)))
                        stage4_dis *= 0.9

                else:
                    controller.reset()
                if manual_active:
                    command = manual_command
                # Overlay the active control mode, marker position, and RC command.
                status = ('MANUAL' if manual_active else
                          'TRACKING' if tracking and drone.is_flying else 'PAUSED')
                lines = [f'{status} | ID: {target_id} | target Z: {TARGET_DISTANCE:.0f} cm',
                         '1: takeoff/track | 2: land | 3/Space: hover | T: track | Q: quit',
                         '4: enable mission45 / wait for marker 4',
                         'Manual priority: W/S A/D Z/X C/V (0.5 s timeout)']
                if m45_state is not None:
                    lines.append(f'mission45: {m45_state} | hold: {hold}/{ALIGN_HOLD_FRAMES}')
                elif mission45_enabled:
                    lines.append('mission45 enabled: waiting for marker 4')
                elif section == 3:
                    lines.append(f'Marker 3 alignment: {stage3_hold}/{ALIGN_HOLD_FRAMES}')
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
            # Stop automatic movement if video has not updated for 0.3 seconds.
            if not manual_active and monotonic() - last_frame_time > 0.3:
                stage3_hold = 0
                align3.reset()
                marker0_missing_since = None
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
        # On quit/error, stop and land, then release video/window/logging resources.
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
