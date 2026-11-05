"""Separate PID banks for guide following and fixed-marker alignment."""
from pyimagesearch.pid import PID


class AlignmentPID:
    def __init__(self):
        # Lateral, forward/back, vertical. Starting gains for flight tuning.
        self.axes = [PID(kP=0.5, kI=0.0, kD=0.05),
                     PID(kP=0.5, kI=0.0, kD=0.05),
                     PID(kP=0.4, kI=0.0, kD=0.05)]
        self.reset()

    def reset(self):
        for pid in self.axes:
            pid.initialize()
        self.ready = False

    def velocity(self, errors, tolerances, limits, enabled=(True, True, True)):
        if not self.ready:
            for pid, error in zip(self.axes, errors):
                pid.initialize()
                pid.prevError = error  # Avoid derivative kick on acquisition.
            self.ready = True
        speeds = []
        for pid, error, tolerance, limit, active in zip(
                self.axes, errors, tolerances, limits, enabled):
            if not active or abs(error) <= tolerance:
                pid.initialize()
                pid.prevError = error
                speeds.append(0)
                continue
            previous_integral = pid.cI
            output = pid.update(error, sleep=0)
            # Do not accumulate integral while pushing into the speed limit.
            if abs(output) > limit and output * error > 0:
                pid.cI = previous_integral
                output = pid.kP * pid.cP + pid.kI * pid.cI + pid.kD * pid.cD
            speeds.append(int(round(max(-limit, min(limit, output)))))
        return tuple(speeds)


class FollowPID(AlignmentPID):
    def __init__(self):
        # Forward/back, vertical, yaw: section 2's requested gains.
        self.axes = [PID(kP=1.0, kI=0.1, kD=0.1),
                     PID(kP=1.0, kI=0.07, kD=0.1),
                     PID(kP=2.0, kI=0.2, kD=0.1)]
        self.reset()
