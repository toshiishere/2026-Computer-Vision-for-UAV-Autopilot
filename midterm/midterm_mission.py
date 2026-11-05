"""Restartable, nonblocking midterm sections. All distances are cm."""
from dataclasses import dataclass, field
from math import atan2, degrees, isfinite
from mission_pid import AlignmentPID, FollowPID

STOP = (0, 0, 0, 0)


@dataclass


class Settings:
    marker_length: float = 15.0
    aruco_y_offset_cm: float = 0.0  # Raw camera Y at the desired physical height.
    marker1_distance: float = 100.0
    marker2_distance: float = 115.0
    follow_distance: float = 100.0
    marker3_distance: float = 100.0
    marker4_distance: float = 150.0
    landing_distance: float = 150.0
    # Practice values: measure these against the actual course before flight.
    under_table_height: float = 35.0
    over_board_height: float = 170.0
    table_pass_distance: float = 150.0
    right_turn_degrees: float = 90.0
    right_clearance_distance: float = 75.0
    travel_speed: int = 20
    tracking_speed: int = 25
    vertical_speed: int = 20
    yaw_speed: int = 25
    position_tolerance: float = 8.0
    height_tolerance: float = 5.0
    yaw_tolerance: float = 5.0
    stable_seconds: float = 0.6
    phase_timeout: float = 25.0
    follow_timeout: float = 120.0
    frame_timeout: float = 0.3
    manual_timeout: float = 0.25

    def validate(self):
        for name, value in vars(self).items():
            if name == 'aruco_y_offset_cm':
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value):
                    raise ValueError(f'{name} must be a finite signed number')
                continue
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be a finite positive number')
        for name in ('travel_speed', 'tracking_speed', 'vertical_speed', 'yaw_speed'):
            if not isinstance(getattr(self, name), int) or getattr(self, name) > 100:
                raise ValueError(f'{name} must be an integer in 1..100')
        if self.under_table_height >= self.over_board_height:
            raise ValueError('under_table_height must be below over_board_height')
        if self.right_turn_degrees >= 180:
            raise ValueError('right_turn_degrees must be below 180')
        for distance in (self.table_pass_distance, self.right_clearance_distance):
            if distance / self.travel_speed >= self.phase_timeout:
                raise ValueError('phase_timeout must exceed every timed travel duration')


@dataclass
class Observation:
    markers: dict = field(default_factory=dict)  # ID -> (x, corrected y, z), cm
    height: float = 0.0
    yaw: float = 0.0


class Mission:
    """One tick per fresh video frame; no sleeps or blocking flight commands."""

    NAMES = ('obstacles 1/2', 'follow marker 0', 'under table / turn / rise',
             'left over board', 'marker 5 / land')

    def __init__(self, settings=None):
        self.cfg = settings or Settings()
        self.cfg.validate()
        self.section = 1
        self.alignment_pid = AlignmentPID()
        self.follow_pid = FollowPID()
        self.restart()

    def _reset_pid(self):
        self.alignment_pid.reset()
        self.follow_pid.reset()

    def restart(self, section=None):
        if section is not None:
            if section not in range(1, 6):
                raise ValueError('Section must be 1..5')
            self.section = section
        self._reset_pid()
        self.phase = ('approach1', 'follow0', 'approach3', 'left', 'align5')[self.section - 1]
        self.elapsed = 0.0
        self.stable = 0.0
        self.last_time = None
        self.turn_start = None
        self.running = False
        self.completed = False
        self.land_requested = False
        self.note = 'Ready; T to start'

    def pause(self, reason='Paused; T to resume, R to restart section'):
        self._reset_pid()
        self.running = False
        self.last_time = None
        self.stable = 0.0
        self.note = reason

    def resume(self):
        if not self.completed and not self.land_requested:
            self._reset_pid()
            self.running = True
            self.last_time = None
            self.note = 'Running'

    def next_section(self):
        self.restart(min(5, self.section + 1))

    def _phase(self, name):
        self._reset_pid()
        self.phase = name
        self.elapsed = self.stable = 0.0

    def _advance(self):
        if self.section == 5:
            self.completed = True
            self.pause('Complete')
        else:
            self.restart(self.section + 1)
            self.resume()

    def _held(self, condition, dt):
        self.stable = self.stable + dt if condition else 0.0
        return self.stable >= self.cfg.stable_seconds

    def _position(self, obs, marker_id):
        p = obs.markers.get(marker_id)
        if p is None or len(p) != 3 or not all(isfinite(v) for v in p) or p[2] <= 0:
            return None
        return p

    def _track(self, obs, marker_id, distance, vertical=True):
        """Strafe to center; preserve the course heading rather than chasing IDs."""
        p = self._position(obs, marker_id)
        if p is None:
            self.alignment_pid.reset()
            self.note = f'Marker {marker_id} missing: hover'
            return STOP, False
        self.note = f'Target marker {marker_id}'
        x, y, z = p
        errors = (x, z - distance, -y if vertical else 0)
        limit = self.cfg.tracking_speed
        rc = self.alignment_pid.velocity(
            errors, (self.cfg.position_tolerance,) * 3, (limit,) * 3,
            enabled=(True, True, vertical))
        return (*rc, 0), all(abs(e) <= self.cfg.position_tolerance for e in errors)

    def _height(self, obs, target):
        if not isfinite(obs.height) or obs.height <= 0:
            self.pause('Height telemetry unavailable; inspect then T to retry')
            return STOP, False
        error = target - obs.height
        reached = abs(error) <= self.cfg.height_tolerance
        return (0, 0, 0 if reached else self.cfg.vertical_speed * (1 if error > 0 else -1), 0), reached

    def tick(self, obs, now):
        if not self.running:
            return STOP
        dt = 0.0 if self.last_time is None else max(0.0, now - self.last_time)
        self.last_time = now
        if dt > self.cfg.frame_timeout:
            self.pause('Video/control gap: hover; T to resume')
            return STOP
        self.elapsed += dt
        timeout = self.cfg.follow_timeout if self.section == 2 else self.cfg.phase_timeout
        if self.elapsed > timeout:
            self.pause('Section phase timed out; reposition and R to restart')
            return STOP
        return (self.section1, self.section2, self.section3,
                self.section4, self.section5)[self.section - 1](obs, dt)

    def section1(self, obs, dt):
        """第一段: marker 1 -> right; marker 2 at 115 cm -> left to 0."""
        if self.phase == 'approach1':
            rc, aligned = self._track(obs, 1, self.cfg.marker1_distance)
            if self._held(aligned, dt):
                self._phase('right')
            return rc
        if self.phase == 'right':
            cleared = self.elapsed >= self.cfg.right_clearance_distance / self.cfg.travel_speed
            if self._held(cleared and self._position(obs, 2) is not None, dt):
                self._phase('approach2')
                return STOP
            return (self.cfg.travel_speed, 0, 0, 0)
        if self.phase == 'approach2':
            rc, aligned = self._track(obs, 2, self.cfg.marker2_distance)
            if self._held(aligned, dt):
                self._phase('left')
            return rc
        if self._held(self._position(obs, 0) is not None, dt):
            self._advance()
            return STOP
        return (-self.cfg.travel_speed, 0, 0, 0)

    def section2(self, obs, dt):
        """第二段: follow marker 0 until ID 2 is absent and ID 3 is visible."""
        position = self._position(obs, 0)
        if position is None:
            self.follow_pid.reset()
            self.note = 'Marker 0 missing: hover'
            rc = STOP
        else:
            self.note = 'Target marker 0'
            heading = degrees(atan2(position[0], position[2]))
            forward, vertical, yaw = self.follow_pid.velocity(
                (position[2] - self.cfg.follow_distance, -position[1], heading),
                (self.cfg.position_tolerance, self.cfg.position_tolerance,
                 self.cfg.yaw_tolerance),
                (self.cfg.tracking_speed, self.cfg.tracking_speed, self.cfg.yaw_speed),
                enabled=(abs(heading) <= 15, True, True))
            # Turn with the guide around the chair; face them before advancing.
            rc = (0, forward, vertical, yaw)
        if self._held(2 not in obs.markers and self._position(obs, 3) is not None, dt):
            self._advance()
            return STOP
        return rc

    def section3(self, obs, dt):
        """第三段: acquire 3, lower, pass 150 cm, turn right, rise, acquire 4."""
        if self.phase == 'approach3':
            rc, aligned = self._track(obs, 3, self.cfg.marker3_distance)
            if self._held(aligned, dt):
                self._phase('lower')
            return rc
        if self.phase == 'lower':
            rc, reached = self._height(obs, self.cfg.under_table_height)
            if self._held(reached, dt):
                self._phase('pass')
            return rc
        if self.phase == 'pass':
            if self.elapsed >= self.cfg.table_pass_distance / self.cfg.travel_speed:
                self.turn_start = None
                self._phase('turn')
                return STOP
            return (0, self.cfg.travel_speed, 0, 0)
        if self.phase == 'turn':
            if not isfinite(obs.yaw):
                self.pause('Yaw telemetry unavailable')
                return STOP
            if self.turn_start is None:
                self.turn_start = obs.yaw
            turned = (obs.yaw - self.turn_start + 180) % 360 - 180
            error = self.cfg.right_turn_degrees - turned
            reached = abs(error) <= self.cfg.yaw_tolerance
            if self._held(reached, dt):
                self._phase('rise')
            return (0, 0, 0, 0 if reached else self.cfg.yaw_speed * (1 if error > 0 else -1))
        if self.phase == 'rise':
            rc, reached = self._height(obs, self.cfg.over_board_height)
            if self._held(reached, dt):
                self._phase('acquire4')
            return rc
        rc, aligned = self._track(obs, 4, self.cfg.marker4_distance, vertical=False)
        height_rc, high = self._height(obs, self.cfg.over_board_height)
        if self._held(aligned and high, dt):
            self._advance()
        return (rc[0], rc[1], height_rc[2], 0) if self.running else STOP

    def section4(self, obs, dt):
        """第四段: keep board clearance and slide left until marker 5."""
        rc, high = self._height(obs, self.cfg.over_board_height)
        if not self.running:
            return STOP
        if self._held(high and self._position(obs, 5) is not None, dt):
            self._advance()
            return STOP
        return (-self.cfg.travel_speed if high else 0, 0, rc[2], 0)

    def section5(self, obs, dt):
        """第五段: align with marker 5 at 150 cm, then request table landing."""
        rc, aligned = self._track(obs, 5, self.cfg.landing_distance)
        if self._held(aligned, dt):
            self.land_requested = True
            self.pause('Marker 5 at 150 cm: landing')
            return STOP
        return rc
