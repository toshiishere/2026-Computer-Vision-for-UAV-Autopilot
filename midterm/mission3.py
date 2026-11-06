"""Task 3 standalone: see ID 3 -> down 50 cm -> forward 150 cm -> turn right 90 deg -> rise until ID 4.

1 takeoff (paused) | T run/resume | R restart task | 2 land | 3/Space pause
W/S forward/back; A/D left/right; Z/X up/down; C/V yaw (pauses the task)
Q/Esc quit and land.
Down / forward / turn use blocking SDK commands: the window and keys freeze
for a few seconds while each one runs.
"""
import logging
from math import copysign, isfinite
from pathlib import Path
from time import monotonic

MARKER_LENGTH = 15.0   # Physical printed marker side length, cm.
ID3 = 3                # Start trigger marker.
ID4 = 4                # Rise until this marker is visible.

ALIGN_TO_ID3 = True    # True: center on ID 3 at ID3_DISTANCE before descending. False: descend as soon as ID 3 is seen.
ID3_DISTANCE = 100.0   # Z distance to hold from ID 3 while aligning, cm.
DESCEND_CM = 50        # SDK 'down' distance (Tello limit 20~500).
FORWARD_CM = 250       # SDK 'forward' distance (Tello limit 20~500).
TURN_DEG = 90          # SDK 'cw' angle (clockwise = turn right).
SDK_SPEED = 30         # SDK move speed in cm/s (Tello limit 10~100).

RISE_SPEED = 20        # RC up speed while looking for ID 4.
MAX_HEIGHT_CM = 170    # Stop rising at this height even if ID 4 was not seen.
RISE_TIMEOUT = 12.0    # Stop rising after this many seconds.
SEE_SECONDS = 0.3      # Marker must stay visible this long to count as seen.

TRACK_SPEED = 25       # Max RC speed while aligning to ID 3.
MIN_SPEED = 10         # Smallest RC value that actually moves a Tello.
POSITION_TOL = 8.0     # Alignment tolerance, cm.
STABLE_SECONDS = 0.6   # Alignment must hold this long.
PHASE_TIMEOUT = 25.0   # SEARCH / ALIGN3 give up after this many seconds.
FRAME_TIMEOUT = 0.3    # No fresh video for this long -> pause and hover.
MANUAL_TIMEOUT = 0.25  # Seconds after the last movement key event.
STOP = (0, 0, 0, 0)

MOVEMENT = {
    ord('w'): (0, 40, 0, 0), ord('s'): (0, -40, 0, 0),
    ord('a'): (-40, 0, 0, 0), ord('d'): (40, 0, 0, 0),
    ord('z'): (0, 0, 50, 0), ord('x'): (0, 0, -50, 0),
    ord('c'): (0, 0, 0, 30), ord('v'): (0, 0, 0, -30),
}

# (phase name, SDK command, value): blocking steps run in this order.
STEPS = (('DESCEND', 'down', DESCEND_CM),
         ('FORWARD', 'forward', FORWARD_CM),
         ('TURN', 'cw', TURN_DEG))


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


def sdk_move(drone, command, value):
    """Run one blocking SDK move ('down', 'forward', 'cw'); raises on failure."""
    drone.send_rc_control(*STOP)
    value = int(value)
    try:
        # Long timeout so djitellopy does not resend the command and overshoot.
        drone.send_control_command(f'{command} {value}', timeout=20)
    except TypeError:
        # Older djitellopy has no timeout parameter.
        {'down': drone.move_down, 'forward': drone.move_forward,
         'cw': drone.rotate_clockwise}[command](value)


class Task3:
    """State machine: SEARCH -> [ALIGN3] -> DESCEND -> FORWARD -> TURN -> RISE -> DONE."""

    def __init__(self):
        self.restart()

    def restart(self):
        self.phase = 'SEARCH'
        self.step_index = 0
        self.running = False
        self.phase_start = None
        self.stable_since = None
        self.note = 'Ready; T to start'

    def pause(self, reason='Paused; T to resume, R to restart'):
        self.running = False
        self.stable_since = None
        self.note = reason

    def resume(self):
        if self.phase != 'DONE':
            self.running = True
            self.phase_start = None  # Restart the phase timer.
            self.note = 'Running'

    def _set_phase(self, name, now):
        self.phase = name
        self.phase_start = now
        self.stable_since = None

    def _held(self, condition, now, seconds):
        if not condition:
            self.stable_since = None
            return False
        if self.stable_since is None:
            self.stable_since = now
        return now - self.stable_since >= seconds

    def _begin_steps(self, now):
        self.step_index = 0
        self._set_phase(STEPS[0][0], now)

    def pending_step(self):
        """The blocking SDK step the main loop must run now, or None."""
        if self.running and self.phase in [s[0] for s in STEPS]:
            return STEPS[self.step_index]
        return None

    def step_done(self, now):
        self.step_index += 1
        if self.step_index < len(STEPS):
            self._set_phase(STEPS[self.step_index][0], now)
        else:
            self._set_phase('RISE', now)

    def _align(self, pos):
        """Strafe to center on ID 3 and hold ID3_DISTANCE (no yaw, no vertical)."""
        x, _, z = pos
        errors = (x, z - ID3_DISTANCE)
        rc = []
        for error in errors:
            if abs(error) <= POSITION_TOL:
                rc.append(0)
                continue
            value = max(-TRACK_SPEED, min(TRACK_SPEED, error * 0.5))
            if abs(value) < MIN_SPEED:
                value = copysign(MIN_SPEED, error)
            rc.append(int(round(value)))
        aligned = all(abs(e) <= POSITION_TOL for e in errors)
        return (rc[0], rc[1], 0, 0), aligned

    def tick(self, seen, pos3, height, now):
        """seen: set of visible marker IDs; pos3: (x, y, z) of ID 3 or None."""
        if not self.running:
            return STOP
        if self.phase_start is None:
            self.phase_start = now
        elapsed = now - self.phase_start

        if self.phase in ('SEARCH', 'ALIGN3') and elapsed > PHASE_TIMEOUT:
            self.pause(f'{self.phase} timed out; reposition then R to restart')
            return STOP

        if self.phase == 'SEARCH':
            self.note = 'Looking for ID 3 (hover)'
            if self._held(ID3 in seen, now, SEE_SECONDS):
                if ALIGN_TO_ID3:
                    self._set_phase('ALIGN3', now)
                else:
                    self._begin_steps(now)
            return STOP

        if self.phase == 'ALIGN3':
            if pos3 is None:
                self.note = 'ID 3 missing: hover'
                self.stable_since = None
                return STOP
            self.note = 'Aligning to ID 3'
            rc, aligned = self._align(pos3)
            if self._held(aligned, now, STABLE_SECONDS):
                self._begin_steps(now)
                return STOP
            return rc

        if self.phase in [s[0] for s in STEPS]:
            self.note = f'{self.phase} (SDK command)'
            return STOP  # The main loop runs the blocking step.

        if self.phase == 'RISE':
            if ID4 in seen:
                self.note = 'ID 4 in view: holding'
                if self._held(True, now, SEE_SECONDS):
                    self._set_phase('DONE', now)
                    self.running = False
                    self.note = 'Done: ID 4 reached (hover; 2 to land)'
                return STOP
            self.stable_since = None
            if isfinite(height) and height >= MAX_HEIGHT_CM:
                self._set_phase('DONE', now)
                self.running = False
                self.note = f'Max height {MAX_HEIGHT_CM} cm reached, ID 4 not seen (hover)'
                return STOP
            if elapsed > RISE_TIMEOUT:
                self.pause('Rise timed out, ID 4 not seen; R to restart')
                return STOP
            self.note = 'Rising until ID 4'
            return (0, 0, RISE_SPEED, 0)

        return STOP  # DONE


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

    task = Task3()
    previous_frame = None
    last_frame_time = monotonic()
    command = STOP
    manual_command = STOP
    manual_until = 0.0
    window = 'Tello Task 3 - focus here for keys'
    drone = Tello()
    rc_filter = RCLogFilter()
    drone.LOGGER.addFilter(rc_filter)
    try:
        drone.connect()
        drone.set_speed(SDK_SPEED)
        print(f'Battery: {drone.get_battery()}%\n{__doc__}')
        drone.streamon()
        reader = drone.get_frame_read()
        cv2.namedWindow(window)
        while True:
            key = normalize_key(cv2.waitKey(20))
            now = monotonic()
            if key in (ord('q'), 27):
                break
            if key in MOVEMENT:
                task.pause('Manual takeover; T to resume, R to restart')
                if drone.is_flying:
                    manual_command = MOVEMENT[key]
                    manual_until = now + MANUAL_TIMEOUT
                command = STOP
            elif key in (ord('1'), ord('2'), ord('3'), ord(' '), ord('r'), ord('t')):
                manual_until = 0.0
                command = STOP
                if drone.is_flying:
                    drone.send_rc_control(*STOP)  # Stop before any blocking takeoff/land.
                if key == ord('1'):
                    task.pause()
                    if not drone.is_flying:
                        drone.takeoff()
                elif key == ord('2'):
                    task.pause()
                    if drone.is_flying:
                        drone.land()
                    task.restart()
                elif key in (ord('3'), ord(' ')):
                    task.pause()
                elif key == ord('r'):
                    task.restart()
                elif key == ord('t') and drone.is_flying:
                    task.resume()

            source = reader.frame
            fresh = source is not None and source.size and source is not previous_frame
            if fresh:
                previous_frame = source
                last_frame_time = now
                frame = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)  # djitellopy gives RGB.
                if detector is not None:
                    corners, ids, _ = detector.detectMarkers(frame)
                else:
                    corners, ids, _ = aruco.detectMarkers(frame, dictionary, parameters=parameters)
                seen = set()
                pos3 = None
                if ids is not None and len(corners):
                    aruco.drawDetectedMarkers(frame, corners, ids)
                    for corner, marker_id in zip(corners, ids.flatten()):
                        marker_id = int(marker_id)
                        seen.add(marker_id)
                        if marker_id == ID3:
                            rvecs, tvecs, _ = aruco.estimatePoseSingleMarkers(
                                [corner], MARKER_LENGTH, intrinsic, distortion)
                            p = tuple(float(v) for v in tvecs[0].reshape(3))
                            if all(isfinite(v) for v in p) and p[2] > 0:
                                if pos3 is None or p[2] < pos3[2]:
                                    pos3 = p
                                cv2.drawFrameAxes(frame, intrinsic, distortion,
                                                  rvecs[0], tvecs[0], MARKER_LENGTH / 2)
                height = float(drone.get_height()) if drone.is_flying else 0.0
                command = task.tick(seen, pos3, height, now) if drone.is_flying else STOP

                lines = [f'TASK3 {"RUNNING" if task.running else "PAUSED"} | phase: {task.phase}',
                         task.note,
                         '1 takeoff | T run | R restart | 2 land | Space pause | Q quit',
                         f'Visible IDs: {sorted(seen)} | height: {height:.0f} cm',
                         f'RC: {command}']
                if pos3 is not None:
                    lines.append(f'ID 3: X={pos3[0]:.1f} Y={pos3[1]:.1f} Z={pos3[2]:.1f} cm')
                for i, line in enumerate(lines):
                    point = (10, 22 + i * 23)
                    cv2.putText(frame, line, point, cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, line, point, cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(window, frame)

                step = task.pending_step() if drone.is_flying else None
                if step is not None:
                    name, sdk_command, value = step
                    print(f'{name}: {sdk_command} {value}')
                    try:
                        sdk_move(drone, sdk_command, value)
                    except Exception as exc:  # noqa: BLE001
                        task.pause(f'{name} failed: {exc}; R to restart')
                        print(task.note)
                    else:
                        task.step_done(monotonic())
                    command = STOP
                    last_frame_time = monotonic()  # Blocking call stalled the loop; not a camera fault.

            if now - last_frame_time > FRAME_TIMEOUT and task.running:
                task.pause('Camera stale: hover; restore video then T to resume')
                command = STOP
                print(task.note)
            if not task.running:
                command = manual_command if now < manual_until else STOP
            if drone.is_flying:
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
                cv2.destroyAllWindows()
                drone.LOGGER.removeFilter(rc_filter)


if __name__ == '__main__':
    main()