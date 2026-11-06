"""獨立程式:對準 marker 4 後,向左平移直到看到 marker 5,然後懸停。

按鍵:
  1:起飛並開始任務   2:降落   3 或 空白鍵:懸停(暫停任務)   Q / Esc:結束 (自動降落)
  【手動優先】W/S:前進/後退  A/D:左/右  Z/X:上/下  C/V:順/逆時針旋轉
  按住手動鍵時,自動控制 (PID / 平移) 會暫時被覆蓋;
  最後一次按鍵後 MANUAL_TIMEOUT 秒內仍維持手動,逾時才交還給自動任務。

流程 (狀態機):
  ALIGN_4     用 PID 對準 marker 4 (左右置中 + 保持 TARGET_DISTANCE)
  STRAFE_LEFT 對準後,固定速度向左平移,直到畫面中出現 marker 5
  ALIGN_5     看到 marker 5 -> 煞車,再用 PID 把 marker 5 置中 (x≈0) 並保持距離
  DONE        marker 5 置中完成 -> 持續懸停

畫面會顯示目標 marker 的 X / Y / Z、距離與轉向誤差 (同原本的 pid_control.py)。

需要的檔案: camera.xml (相機校正檔,需含 intrinsic / distortion)
            pyimagesearch/pid.py (你原本就有的 PID 類別)
"""
from math import atan2, degrees, isfinite, sqrt
import logging
from pathlib import Path
from time import monotonic

from pyimagesearch.pid import PID

# ======================= 參數區 (需實飛微調) =======================
MARKER_LENGTH = 15.0     # 標記實際邊長 (cm),用來估算真實距離
TARGET_DISTANCE = 100.0  # 對準 marker 4 時,與標記保持的距離 (cm)
DEADBAND = 3.0           # 位置誤差容許範圍 (cm),在範圍內就不動
MAX_SPEED = 25           # PID 輸出的 RC 速度上限
STRAFE_SPEED = 20        # 向左平移速度 (RC 值,實際送 -20;負 = 向左)
STRAFE_TIMEOUT = 20.0    # 安全機制:平移超過這個秒數還沒看到 marker 5 就放棄
ALIGN_HOLD_FRAMES = 5    # 需「連續」N 幀都在誤差內,才算真的對準
STALE_FRAME_LIMIT = 0.3  # 影像超過這個秒數沒更新就強制停止 (安全機制)
STOP = (0, 0, 0, 0)      # RC 指令格式: (左右, 前後, 上下, 偏航)

# ---- 手動控制 (優先於自動控制) ----
MANUAL_TIMEOUT = 0.5     # 最後一次按下移動鍵後,手動優先維持的秒數
MOVEMENT = {             # 按鍵 -> RC 指令 (左右, 前後, 上下, 偏航)
    ord('w'): (0, 40, 0, 0), ord('s'): (0, -40, 0, 0),    # 前進 / 後退
    ord('a'): (-40, 0, 0, 0), ord('d'): (40, 0, 0, 0),    # 向左 / 向右
    ord('z'): (0, 0, 50, 0), ord('x'): (0, 0, -50, 0),    # 上升 / 下降
    ord('c'): (0, 0, 0, 30), ord('v'): (0, 0, 0, -30),    # 順時針 / 逆時針
}


def normalize_key(key):
    """把 cv2.waitKey 的回傳值轉成小寫 ASCII (大小寫鎖定也能用);沒按鍵回傳 -1。"""
    if key < 0:
        return -1
    key &= 0xFF
    return key + 32 if ord('A') <= key <= ord('Z') else key

ID_START = 4             # 先對準的 marker
ID_END = 5               # 向左平移要找的 marker

# 狀態名稱
ALIGN_4, STRAFE_LEFT, ALIGN_5, DONE = 'ALIGN_4', 'STRAFE_LEFT', 'ALIGN_5', 'DONE'


class RCLogFilter(logging.Filter):
    """過濾掉高頻率的 rc 指令 log,避免洗版。"""

    def filter(self, record):
        return not (
            record.levelno == logging.INFO
            and record.getMessage().startswith(
                "Send command (no response expected): 'rc "
            )
        )


class AlignController:
    """用 2 個 PID 對準標記:左右 (x) 與前後 (z)。"""

    def __init__(self):
        self.axes = [PID(kP=1.0, kI=0.1, kD=0.1),    # 左右
                     PID(kP=1.0, kI=0.07, kD=0.1)]   # 前後
        self.reset()

    def reset(self):
        # 清除積分/微分記憶,下次重新取得第一筆誤差
        for pid in self.axes:
            pid.initialize()
        self.ready = False

    def velocity(self, position):
        """
        position: (x, y, z) 標記在相機座標系的位置 (x 向右、z 向前,cm)
        回傳: (rc指令, 是否已在誤差範圍內)
        """
        x, _, z = map(float, position)
        if not all(isfinite(v) for v in (x, z)) or z <= 0:
            self.reset()
            return STOP, False

        # 誤差:x 為正代表標記在右邊 -> 需要向右;z 大於目標 -> 需要前進
        errors = (x, z - TARGET_DISTANCE)

        if not self.ready:
            # 第一次取得誤差時,把 prevError 設成目前誤差,避免微分項暴衝
            for pid, error in zip(self.axes, errors):
                pid.initialize()
                pid.prevError = error
            self.ready = True

        speeds = []
        for pid, error in zip(self.axes, errors):
            output = pid.update(error, sleep=0)
            # 誤差在容許範圍內輸出 0,否則限幅在 ±MAX_SPEED
            speeds.append(0 if abs(error) <= DEADBAND else
                          int(round(max(-MAX_SPEED, min(MAX_SPEED, output)))))

        in_tolerance = all(abs(e) <= DEADBAND for e in errors)
        lr, fb = speeds
        return (lr, fb, 0, 0), in_tolerance   # 不控制上下與偏航


def main():
    import cv2
    from djitellopy import Tello

    # ---------- 讀取相機校正檔 ----------
    calibration = Path(__file__).resolve().with_name('camera.xml')
    fs = cv2.FileStorage(str(calibration), cv2.FILE_STORAGE_READ)
    try:
        if not fs.isOpened():
            raise RuntimeError(f'Cannot open camera calibration: {calibration}')
        intrinsic = fs.getNode('intrinsic').mat()
        distortion = fs.getNode('distortion').mat()
        if intrinsic is None or distortion is None:
            raise RuntimeError('camera.xml 需要 intrinsic 與 distortion')
    finally:
        fs.release()

    # ---------- 建立 ArUco 偵測器 (相容新舊版 OpenCV) ----------
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
    parameters = (aruco.DetectorParameters() if hasattr(aruco, 'DetectorParameters')
                  else aruco.DetectorParameters_create())
    detector = (aruco.ArucoDetector(dictionary, parameters)
                if hasattr(aruco, 'ArucoDetector') else None)

    controller = AlignController()
    running = False            # 是否正在執行任務 (按 1 後為 True)
    state = ALIGN_4            # 狀態機目前狀態
    hold = 0                   # 已連續對準的幀數
    strafe_start = 0.0         # 開始平移的時間 (判斷超時用)
    previous_frame = None
    last_frame_time = monotonic()
    command = STOP
    manual_command = STOP      # 最近一次手動按鍵對應的 RC 指令
    manual_until = 0.0         # 手動優先的截止時間 (monotonic 秒)
    was_manual = False         # 上一輪是否處於手動模式 (用來偵測「剛結束手動」)

    window = 'Tello marker4 -> marker5'
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
            # ---------- 1. 鍵盤 ----------
            key = normalize_key(cv2.waitKey(30))
            if key in (ord('q'), 27):
                break
            if key in MOVEMENT:                       # 手動移動鍵 (W/S/A/D/Z/X/C/V)
                if drone.is_flying:
                    manual_command = MOVEMENT[key]
                    manual_until = monotonic() + MANUAL_TIMEOUT   # 每次按鍵都延長手動時間
                    controller.reset()                # 清掉 PID 記憶,交還控制權時重新開始
                    hold = 0
            elif key == ord('1'):                     # 起飛並從頭開始任務
                if not drone.is_flying:
                    drone.takeoff()
                manual_until = 0.0                    # 取消手動狀態
                running, state, hold = True, ALIGN_4, 0
                command = STOP
                controller.reset()
            elif key == ord('2'):                     # 降落
                manual_until = 0.0
                running = False
                command = STOP
                if drone.is_flying:
                    drone.send_rc_control(*STOP)
                    drone.land()
                controller.reset()
            elif key in (ord('3'), ord(' ')):         # 暫停任務並懸停
                manual_until = 0.0
                running = False
                command = STOP
                if drone.is_flying:
                    drone.send_rc_control(*STOP)
                controller.reset()

            # 手動是否生效:正在飛行,且還在最後一次按鍵後的 MANUAL_TIMEOUT 秒內
            manual_active = drone.is_flying and monotonic() < manual_until
            if manual_active and state == STRAFE_LEFT:
                strafe_start = monotonic()            # 手動期間不計入平移超時

            # ---------- 2. 取得新影像 ----------
            source = frame_read.frame
            fresh = source is not None and source.size and source is not previous_frame

            if fresh:
                previous_frame = source
                last_frame_time = monotonic()
                # DJITelloPy 給的是 RGB,OpenCV 要 BGR
                frame = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)

                # ---------- 3. 偵測所有 marker,記錄 {id: (x, y, z)} ----------
                positions = {}
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(frame)
                else:
                    corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)
                if ids is not None and len(corners):
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    for index, marker_id in enumerate(ids.flatten()):
                        rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                            [corners[index]], MARKER_LENGTH, intrinsic, distortion)
                        pos = tuple(float(v) for v in tvecs[0].reshape(3))
                        if not all(isfinite(v) for v in pos) or pos[2] <= 0:
                            continue  # 姿態無效就略過這個 marker
                        positions[int(marker_id)] = pos
                        cv2.drawFrameAxes(frame, intrinsic, distortion,
                                          rvecs[0], tvecs[0], MARKER_LENGTH / 2)

                # ---------- 4. 狀態機:決定這一幀的 RC 指令 ----------
                command = STOP
                if manual_active:
                    # 手動優先:本幀不執行自動任務,狀態機暫停 (狀態保留,不會重來)
                    command = manual_command
                    controller.reset()
                    hold = 0
                elif running and drone.is_flying:

                    # --- 狀態 1:對準 marker 4 ---
                    if state == ALIGN_4:
                        if ID_START in positions:
                            command, ok = controller.velocity(positions[ID_START])
                            hold = hold + 1 if ok else 0   # 連續計數,一幀不合格就歸零
                            if hold >= ALIGN_HOLD_FRAMES:
                                state = STRAFE_LEFT
                                strafe_start = monotonic()
                                controller.reset()
                                command = STOP
                                print('已對準 marker 4 -> 開始向左平移')
                        else:
                            # 看不到 marker 4:懸停等待
                            controller.reset()
                            hold = 0

                    # --- 狀態 2:向左平移直到看到 marker 5 ---
                    elif state == STRAFE_LEFT:
                        if ID_END in positions:
                            state = ALIGN_5                # 看到了!先煞車,下一幀開始置中
                            hold = 0
                            controller.reset()
                            command = STOP
                            print('看到 marker 5 -> 開始置中')
                        elif monotonic() - strafe_start > STRAFE_TIMEOUT:
                            state = DONE                   # 超時保護
                            command = STOP
                            print('平移超時,沒看到 marker 5 -> 懸停')
                        else:
                            # 左右軸負值 = 向左;其餘軸為 0
                            command = (-STRAFE_SPEED, 0, 0, 0)

                    # --- 狀態 3:把 marker 5 置中 ---
                    elif state == ALIGN_5:
                        if ID_END in positions:
                            # PID 讓 marker 5 的 x 趨近 0 (= 畫面水平正中央),
                            # 同時保持與標記 TARGET_DISTANCE 的距離
                            command, ok = controller.velocity(positions[ID_END])
                            hold = hold + 1 if ok else 0
                            if hold >= ALIGN_HOLD_FRAMES:
                                state = DONE
                                command = STOP
                                print('marker 5 已置中 -> 完成')
                        else:
                            # 置中途中暫時看不到 marker 5:原地懸停等它回來
                            # (不退回向左平移,避免越飛越遠;需要時可用手動修正)
                            controller.reset()
                            hold = 0

                    # --- 狀態 4:完成,持續懸停 ---
                    else:
                        command = STOP

                # ---------- 5. 在畫面上顯示資訊 ----------
                mode = 'MANUAL' if manual_active else ('RUNNING' if running else 'PAUSED')
                lines = [f'{mode} | state: {state}',
                         '1: start | 2: land | 3/Space: hover | Q: quit | W/S A/D Z/X C/V: manual',
                         f'seen IDs: {sorted(positions)} | hold: {hold}/{ALIGN_HOLD_FRAMES}']

                # 顯示「目前目標 marker」的位置:ALIGN_4 階段看 marker 4,之後看 marker 5
                show_id = ID_START if state == ALIGN_4 else ID_END
                lines.append(f'target ID: {show_id} | target Z: {TARGET_DISTANCE:.0f} cm')
                target_pos = positions.get(show_id)
                if target_pos is None:
                    lines.append(f'Target {show_id} missing')
                else:
                    x, y, z = target_pos
                    distance = sqrt(x*x + y*y + z*z)   # 相機到標記的直線距離
                    lines.append(f'X={x:.1f} Y={y:.1f} Z={z:.1f} cm | distance={distance:.1f} cm')
                    # 轉向誤差:標記偏離相機正前方的水平角度 (0 = 正對、正值 = 在右邊)
                    lines.append(f'Turn error: {degrees(atan2(x, z)):.1f} deg')
                lines.append(f'RC: {command}')
                for i, line in enumerate(lines):
                    pt = (10, 25 + 25 * i)
                    cv2.putText(frame, line, pt, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, line, pt, cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(window, frame)

            # ---------- 6. 安全機制 ----------
            if manual_active:
                command = manual_command              # 手動永遠優先 (沒新影像時也照送)
            elif was_manual:
                command = STOP                        # 手動剛逾時:先停一下,避免沿用舊的手動指令
            elif not running:
                command = STOP                        # 沒有執行任務時一律停止
            was_manual = manual_active
            # 影像太久沒更新 (斷訊/卡住) -> 自動模式立即停止;手動模式不受影響,
            # 因為手動不依賴影像,這樣即使影像卡住你仍能手動操控
            # 注意:沒有新影像的迴圈會「沿用上一個指令」,讓平移更平順
            if not manual_active and monotonic() - last_frame_time > STALE_FRAME_LIMIT:
                command = STOP
                controller.reset()

            # ---------- 7. 持續送出 RC 指令 (懸停時也要送) ----------
            if drone.is_flying:
                drone.send_rc_control(*command)

            if cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                break

    except KeyboardInterrupt:
        pass
    finally:
        # 無論如何結束,都先停止、降落、釋放資源
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