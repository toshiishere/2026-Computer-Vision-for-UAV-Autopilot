"""Midterm flight runner. Focus the video window for keyboard control.

1 takeoff (paused); 2 land; 3/Space pause; T run/resume; R restart section.
4/5/6/7/8 select sections 1/2/3/4/5 (paused); N next section (paused).
After T, completed sections continue automatically through section 5 landing.
W/S forward/back; A/D left/right; Z/X up/down; C/V yaw (pauses mission).
H height; B battery; Q/Esc quit and land. See README.md before practice.
"""
import argparse
import json
from pathlib import Path
from time import monotonic

from keyboard_djitellopy import MOVEMENT, normalize_key
from midterm_mission import Mission, Observation, Settings, STOP
from pid_control import RCLogFilter


def load_settings(path):
    cfg = Settings(**json.loads(Path(path).read_text(encoding='utf-8'))) if path else Settings()
    cfg.validate()
    return cfg


class Vision:
    def __init__(self, cv2, cfg):
        self.cv2, self.cfg = cv2, cfg
        self.raw_markers = {}
        path = Path(__file__).with_name('camera.xml')
        fs = cv2.FileStorage(str(path), cv2.FILE_STORAGE_READ)
        try:
            if not fs.isOpened():
                raise RuntimeError(f'Cannot open calibration: {path}')
            self.intrinsic = fs.getNode('intrinsic').mat()
            self.distortion = fs.getNode('distortion').mat()
            if self.intrinsic is None or self.distortion is None:
                raise RuntimeError('camera.xml needs intrinsic and distortion matrices')
        finally:
            fs.release()
        aruco = cv2.aruco
        self.dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_250)
        self.parameters = (aruco.DetectorParameters() if hasattr(aruco, 'DetectorParameters')
                           else aruco.DetectorParameters_create())
        self.detector = (aruco.ArucoDetector(self.dictionary, self.parameters)
                         if hasattr(aruco, 'ArucoDetector') else None)

    def calibrate_position(self, position):
        """Camera Y points down. Make the desired raw Y the controller's zero."""
        x, raw_y, z = position
        return (x, raw_y - self.cfg.aruco_y_offset_cm, z)

    def read(self, frame):
        cv2, aruco = self.cv2, self.cv2.aruco
        if self.detector is not None:
            corners, ids, _ = self.detector.detectMarkers(frame)
        else:
            corners, ids, _ = aruco.detectMarkers(frame, self.dictionary, parameters=self.parameters)
        markers = {}
        self.raw_markers = {}
        if ids is not None:
            aruco.drawDetectedMarkers(frame, corners, ids)
            for corner, marker_id in zip(corners, ids.flatten()):
                rotations, positions, _ = aruco.estimatePoseSingleMarkers(
                    [corner], self.cfg.marker_length, self.intrinsic, self.distortion)
                position = tuple(float(v) for v in positions[0].reshape(3))
                marker_id = int(marker_id)
                if marker_id not in markers or position[2] < markers[marker_id][2]:
                    self.raw_markers[marker_id] = position
                    markers[marker_id] = self.calibrate_position(position)
                cv2.drawFrameAxes(frame, self.intrinsic, self.distortion,
                                  rotations[0], positions[0], self.cfg.marker_length / 2)
        return markers

class KeyboardControl:
    """Keyboard policy is independent of vision and testable without a drone."""
    def __init__(self, mission):
        self.mission = mission
        self.manual_until = 0.0
        self.manual_command = STOP
        self.revision = 0

    def handle(self, drone, key, now):
        key = normalize_key(key)
        m = self.mission
        if key in (ord('q'), 27):
            m.pause('Exiting')
            return False
        if key < 0:
            return True
        if key in MOVEMENT:
            self.revision += 1
            m.pause('Manual takeover; T to resume, R to restart')
            if drone.is_flying:
                self.manual_command = MOVEMENT[key]
                self.manual_until = now + m.cfg.manual_timeout
            return True
        if key in (ord('1'), ord('2'), ord('3'), ord(' '), ord('r'), ord('n'), ord('t')) or ord('4') <= key <= ord('8'):
            self.revision += 1
            self.manual_until = 0.0
            # Stop previous automatic movement before a blocking takeoff/land.
            if drone.is_flying:
                drone.send_rc_control(*STOP)
        if key == ord('1'):
            m.pause()
            if not drone.is_flying:
                drone.takeoff()
        elif key == ord('2'):
            m.pause()
            if drone.is_flying:
                drone.land()
            m.restart()
        elif key in (ord('3'), ord(' ')):
            m.pause()
        elif key == ord('r'):
            m.restart()
        elif ord('4') <= key <= ord('8'):
            m.restart(key - ord('4') + 1)
        elif key == ord('n'):
            m.next_section()
        elif key == ord('t') and drone.is_flying:
            m.resume()
        elif key == ord('h'):
            print(f'Height: {drone.get_height()} cm')
        elif key == ord('b'):
            print(f'Battery: {drone.get_battery()}%')
        return True

    def manual_velocity(self, now):
        return self.manual_command if now < self.manual_until else STOP


def run(cfg, section=1):
    import cv2
    from djitellopy import Tello

    vision = Vision(cv2, cfg)
    mission = Mission(cfg)
    mission.restart(section)
    controls = KeyboardControl(mission)
    drone = Tello()
    rc_filter = RCLogFilter()
    drone.LOGGER.addFilter(rc_filter)
    window = 'Tello Midterm - focus here for keys'
    previous_frame = None
    last_frame_time = monotonic()
    command = STOP
    try:
        drone.connect()
        print(f'Battery: {drone.get_battery()}%\n{__doc__}')
        drone.streamon()
        reader = drone.get_frame_read()
        cv2.namedWindow(window)
        while True:
            revision = controls.revision
            if not controls.handle(drone, cv2.waitKey(20), monotonic()):
                break
            if controls.revision != revision:
                command = STOP
            now = monotonic()
            source = reader.frame
            fresh = source is not None and source.size and source is not previous_frame
            if fresh:
                previous_frame = source
                last_frame_time = now
                frame = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)
                markers = vision.read(frame)
                observation = Observation(markers, drone.get_height(), drone.get_yaw())
                command = mission.tick(observation, now) if drone.is_flying else STOP
                if mission.land_requested:
                    drone.send_rc_control(*STOP)
                    if drone.is_flying:
                        drone.land()
                    mission.land_requested = False
                    mission.completed = True
                    mission.note = 'Complete: landed at marker 5'
                status = 'RUNNING' if mission.running else 'PAUSED'
                lines = [f'MIDTERM {status} | section {mission.section}: {mission.NAMES[mission.section-1]}',
                         f'Phase: {mission.phase} ({mission.elapsed:.1f}s) | {mission.note}',
                         '1 takeoff | 2 land | Space pause | T resume | R restart | Q quit',
                         '4..8 sections 1..5 | N next | W/S A/D Z/X C/V manual takeover',
                         f'Visible IDs: {sorted(markers)} | height: {observation.height} cm',
                         f'ArUco Y calibration: corrected Y = raw Y - {cfg.aruco_y_offset_cm:+.1f} cm']
                for marker_id, p in sorted(markers.items()):
                    raw_y = vision.raw_markers[marker_id][1]
                    lines.append(f'ID {marker_id}: X={p[0]:.1f} Yraw={raw_y:.1f} Y={p[1]:.1f} Z={p[2]:.1f} cm')
                for i, line in enumerate(lines):
                    cv2.putText(frame, line, (10, 22 + i * 23), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, line, (10, 22 + i * 23), cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.imshow(window, frame)
            if now - last_frame_time > cfg.frame_timeout and mission.running:
                mission.pause('Camera stale: hover; restore video then T to resume')
                command = STOP
                print(mission.note)
            if not mission.running:
                command = controls.manual_velocity(now)
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', help='JSON overrides for Settings (see midterm_config.json)')
    parser.add_argument('--section', type=int, choices=range(1, 6), default=1)
    args = parser.parse_args()
    run(load_settings(args.config), section=args.section)


if __name__ == '__main__':
    main()
