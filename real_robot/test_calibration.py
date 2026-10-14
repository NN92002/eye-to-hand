import unittest
import cv2
import numpy as np
from core import solve
from solve_lrbo import accept_results


def pose(rv, t):
    result = np.eye(4)
    result[:3, :3] = cv2.Rodrigues(np.array(rv, dtype=float))[0]
    result[:3, 3] = t
    return result


class CalibrationTests(unittest.TestCase):
    def test_eye_to_hand_direction_and_units(self):
        rng = np.random.default_rng(42)
        expected = pose([.2, -.3, .1], [.7, -.4, 1.2])
        tool_target = pose([.1, .2, -.1], [.02, .03, .15])
        robot = [pose(rng.normal(size=3)*.5, rng.uniform(-.5, .5, 3)) for _ in range(20)]
        target = [np.linalg.inv(expected) @ t @ tool_target for t in robot]
        actual, diagnostics = solve(robot, target)
        np.testing.assert_allclose(actual, expected, atol=1e-7)
        self.assertLess(diagnostics['closure_translation_rms_mm'], 1e-6)

    def test_degenerate_motion_rejected(self):
        with self.assertRaisesRegex(ValueError, 'diversity'):
            solve([np.eye(4)]*10, [np.eye(4)]*10)

    def test_lrbo_quality_and_disagreement(self):
        np.testing.assert_allclose(accept_results([np.eye(4)]*2, [.95]*2, [.01]*2), np.eye(4))
        for fitness, rmse in [([.8], [.01]), ([.95], [.03]), ([float('nan')], [.01])]:
            with self.assertRaises(ValueError):
                accept_results([np.eye(4)], fitness, rmse)
        with self.assertRaisesRegex(ValueError, 'disagree'):
            accept_results([np.eye(4), pose([0, 0, 0], [.06, 0, 0])], [.95]*2, [.01]*2)


if __name__ == '__main__':
    unittest.main()
