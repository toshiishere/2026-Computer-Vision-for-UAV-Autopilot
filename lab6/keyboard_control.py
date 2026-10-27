"""Run this file and focus the video window to control the Tello."""
from time import monotonic

MOVEMENT = {
    ord('w'): (0, 40, 0, 0), ord('s'): (0, -40, 0, 0),
    ord('a'): (-40, 0, 0, 0), ord('d'): (40, 0, 0, 0),
    ord('z'): (0, 0, 50, 0), ord('x'): (0, 0, -50, 0),
    ord('c'): (0, 0, 0, 30), ord('v'): (0, 0, 0, -30),
}
STOP = (0, 0, 0, 0)
CONTROLS = (
    '1: takeoff | 2: land | 3/Space: stop | Q/Esc: quit',
    'W/S: forward/back | A/D: left/right | Z/X: up/down',
    'C/V: clockwise/counterclockwise | 5: height | 6: battery',
    'Repeat movement keys; stops after 0.25 s without input.',
)


def normalize_key(key):
    if key < 0:
        return -1
    key &= 0xFF
    return key + 32 if ord('A') <= key <= ord('Z') else key


def keyboard(drone, key):
    """Handle an OpenCV key code; return False to request exit."""
    key = normalize_key(key)
    if key in (ord('q'), 27):
        return False
    if key == ord('1'):
        if not drone.is_flying:
            drone.takeoff()
    elif key == ord('2'):
        if drone.is_flying:
            drone.send_rc_control(*STOP)
            drone.land()
    elif key in (ord('3'), ord(' ')):
        drone.send_rc_control(*STOP)
    elif key in MOVEMENT:
        if drone.is_flying:
            drone.send_rc_control(*MOVEMENT[key])
    elif key == ord('5'):
        print(f'Height: {drone.get_height()} cm')
    elif key == ord('6'):
        print(f'Battery: {drone.get_battery()}%')
    return True


def main():
    import cv2
    from djitellopy import Tello

    window = 'Tello manual control'
    drone = Tello()
    velocity = STOP
    last_movement = 0.0
    try:
        drone.connect()
        print(f'Battery: {drone.get_battery()}%')
        print('\n'.join(CONTROLS))
        drone.streamon()
        frame_read = drone.get_frame_read()
        cv2.namedWindow(window)
        while True:
            source = frame_read.frame
            if source is not None and source.size:
                frame = source.copy()
                for index, line in enumerate(CONTROLS):
                    position = (10, 25 + 25 * index)
                    cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (0, 0, 0), 3, cv2.LINE_AA)
                    cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX,
                                0.45, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.imshow(window, frame)
            key = normalize_key(cv2.waitKey(30))
            if not keyboard(drone, key):
                break
            if key in MOVEMENT and drone.is_flying:
                velocity = MOVEMENT[key]
                last_movement = monotonic()
            elif key in (ord('1'), ord('2'), ord('3'), ord(' ')):
                velocity = STOP
            if monotonic() - last_movement > 0.25 or not drone.is_flying:
                velocity = STOP
            if drone.is_flying:
                drone.send_rc_control(*velocity)
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


if __name__ == '__main__':
    main()
