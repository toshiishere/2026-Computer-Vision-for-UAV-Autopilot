"""Offline checks for section-specific PID behavior; no drone required."""
import unittest
from unittest.mock import patch

from midterm_mission import Mission, Observation, STOP


class MissionPIDTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.clock = patch('pyimagesearch.pid.time.time', side_effect=lambda: self.now)
        self.clock.start()
        self.addCleanup(self.clock.stop)
        self.mission = Mission()
        self.mission.resume()

    def test_section2_uses_requested_gains_and_axis_order(self):
        self.mission.restart(2)
        self.mission.resume()
        self.assertEqual(
            [(p.kP, p.kI, p.kD) for p in self.mission.follow_pid.axes],
            [(1.0, 0.1, 0.1), (1.0, 0.07, 0.1), (2.0, 0.2, 0.1)])
        self.assertEqual(self.mission.tick(
            Observation({0: (20, 10, 120)}), self.now), (0, 20, -10, 19))
        self.assertFalse(self.mission.alignment_pid.ready)

    def test_alignment_uses_strafe_and_conservative_gains(self):
        rc, aligned = self.mission._track(Observation({1: (20, 10, 120)}), 1, 100)
        self.assertEqual(rc, (10, 10, -4, 0))
        self.assertFalse(aligned)
        self.assertFalse(self.mission.follow_pid.ready)
        rc, aligned = self.mission._track(Observation({1: (8, -8, 108)}), 1, 100)
        self.assertEqual(rc, STOP)
        self.assertTrue(aligned)

    def test_marker4_ignores_marker_y(self):
        rc, aligned = self.mission._track(
            Observation({4: (0, 100, 150)}), 4, 150, vertical=False)
        self.assertEqual(rc, STOP)
        self.assertTrue(aligned)

    def test_turn_first_and_speed_limits(self):
        self.mission.restart(2)
        self.mission.resume()
        rc = self.mission.tick(Observation({0: (200, -200, 200)}), self.now)
        self.assertEqual(rc, (0, 0, 25, 25))
        for _ in range(10):
            self.now += 0.1
            self.mission.tick(Observation({0: (200, -200, 200)}), self.now)
        self.assertEqual([p.cI for p in self.mission.follow_pid.axes], [0, 0, 0])

    def test_loss_pause_and_phase_change_reset_state(self):
        self.mission._track(Observation({1: (20, 0, 120)}), 1, 100)
        self.assertEqual(self.mission._track(Observation(), 1, 100), (STOP, False))
        self.assertFalse(self.mission.alignment_pid.ready)
        self.mission._track(Observation({1: (20, 0, 120)}), 1, 100)
        self.mission._phase('right')
        self.assertFalse(self.mission.alignment_pid.ready)
        self.mission.restart(2)
        self.mission.resume()
        self.mission.tick(Observation({0: (20, 10, 120)}), self.now)
        self.assertEqual(self.mission.tick(Observation(), self.now), STOP)
        self.assertFalse(self.mission.follow_pid.ready)
        self.mission.tick(Observation({0: (20, 10, 120)}), self.now)
        self.mission.pause()
        self.assertFalse(self.mission.follow_pid.ready)

    def test_section2_exit_still_requires_held_marker_condition(self):
        self.mission.restart(2)
        self.mission.resume()
        obs = Observation({3: (0, 0, 100)})
        self.mission.tick(obs, self.now)
        for _ in range(7):
            self.now += 0.1
            self.mission.tick(obs, self.now)
        self.assertEqual(self.mission.section, 3)
        self.assertEqual(self.mission.phase, 'approach3')
        self.assertTrue(self.mission.running)


if __name__ == '__main__':
    unittest.main()
