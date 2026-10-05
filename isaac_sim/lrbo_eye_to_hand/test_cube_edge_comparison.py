"""Offline checks for one-shot state and calibration comparison semantics."""
import contextlib
import copy
import io
import tempfile
import queue
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import run_lrbo_cube_edge as cube
from calibration_comparison import compare_calibrations, load_calibration


class OneShotTests(unittest.TestCase):
    def setUp(self):
        self.rgb = np.full((150, 180, 4), 100, np.uint8)
        cv2.rectangle(self.rgb, (30, 25), (145, 120), (0, 240, 0, 255), -1)
        cv2.rectangle(self.rgb, (85, 25), (145, 120), (0, 110, 0, 255), -1)
        depth = np.full((150, 180), 1.4, np.float32)
        depth[25:121, 30:146] = .8
        self.frame = {'rgb': self.rgb, 'distance_to_image_plane': depth, 'rendering_frame': 1}
        camera = SimpleNamespace(get_current_frame=lambda: self.frame,
                                 get_intrinsics_matrix=lambda: np.array([[100.,0,90],[0,100.,75],[0,0,1]]))
        self.detector = cube.CubeEdgeDetector(camera=camera)

    def update(self):
        with contextlib.redirect_stdout(io.StringIO()):
            self.detector.update_detection(self.rgb, show_window=False)

    def test_preview_does_not_detect_and_frozen_result_does_not_change(self):
        with patch.object(cube, 'extract_edges', wraps=cube.extract_edges) as extract:
            self.update()
            extract.assert_not_called()
            self.assertIsNone(self.detector.target_edge)
            self.detector.request_detection()
            self.update()
            self.assertEqual(extract.call_count, 1)
            self.assertIsNotNone(self.detector.target_edge)
            target = copy.deepcopy(self.detector.target_edge)
            display = self.detector.display_frame.copy()
            depth = copy.deepcopy(self.detector.edge_depth)
            self.frame['rgb'] = np.zeros_like(self.rgb)
            self.frame['rendering_frame'] = 2
            self.update()
            self.assertEqual(extract.call_count, 1)
            self.assertEqual(target, self.detector.target_edge)
            self.assertEqual(depth, self.detector.edge_depth)
            np.testing.assert_array_equal(display, self.detector.display_frame)
            self.detector.request_detection()
            self.update()
            self.assertEqual(extract.call_count, 2)
            self.assertIsNone(self.detector.target_edge)  # failed request clears old edge
            self.assertTrue(self.detector._frozen)

    def test_clear_returns_to_live_and_request_can_reuse_current_frame(self):
        self.detector.request_detection()
        self.update()
        self.detector.clear_detection()
        self.assertFalse(self.detector._frozen)
        self.assertIsNone(self.detector.target_edge)
        self.detector.request_detection()
        self.update()
        self.assertIsNotNone(self.detector.target_edge)

    def test_no_calibrator_is_created_and_old_commands_are_discarded(self):
        with patch.object(cube.lrbo, 'LRBOEyeToHandCalibrator') as calibrator:
            detector = cube.CubeEdgeDetector(camera=None, stage=None, robot_base_path='/robot')
            calibrator.assert_not_called()
        commands = queue.Queue()
        commands.put('capture'); commands.put('solve')
        scene = SimpleNamespace(CALIBRATION_COMMAND_QUEUE=commands, queue=queue)
        with patch.object(cube.lrbo, 'SCENE', scene):
            detector.process_commands()
        self.assertTrue(commands.empty())
        detector.close()


class ComparisonTests(unittest.TestCase):
    def record(self, matrix, parent='base', gt=None):
        return {'path': 'example.yaml', 'matrix': matrix, 'ground_truth': gt,
                'metadata': {'parent_frame': parent, 'child_frame': 'camera_color_optical_frame'}}

    def test_known_translation_and_rotation_difference(self):
        a, b = np.eye(4), np.eye(4)
        angle = np.radians(10)
        b[:3, :3] = [[np.cos(angle), -np.sin(angle), 0],
                      [np.sin(angle), np.cos(angle), 0], [0, 0, 1]]
        b[:3, 3] = [.003, .004, 0]
        result = compare_calibrations(self.record(a), self.record(b))
        self.assertAlmostEqual(result['translation_difference_mm'], 5)
        self.assertAlmostEqual(result['rotation_difference_deg'], 10)
        self.assertNotIn('opencv_gt_translation_error_mm', result)

    def test_alias_requires_gt_and_moved_camera_is_rejected(self):
        identity = np.eye(4)
        with self.assertRaisesRegex(ValueError, 'Parent frames'):
            compare_calibrations(self.record(identity), self.record(identity, '/robot'))
        result = compare_calibrations(self.record(identity, gt=identity),
                                      self.record(identity, '/robot', identity))
        self.assertEqual(result['translation_difference_mm'], 0)
        moved = identity.copy(); moved[0, 3] = .1
        with self.assertRaisesRegex(ValueError, 'poses differ'):
            compare_calibrations(self.record(identity, gt=identity),
                                 self.record(identity, '/robot', moved))

    def test_saved_exports_and_companion_npy(self):
        root = Path(cube.__file__).resolve().parent
        opencv = load_calibration(root.parent / 'calibration' / 'eye_to_hand_20261002_143515.yaml')
        saved = load_calibration(root / 'results' / 'lrbo_eye_to_hand_20261002_210826_822078.json')
        result = compare_calibrations(opencv, saved)
        self.assertAlmostEqual(result['translation_difference_mm'], 2.316764, places=4)
        self.assertAlmostEqual(result['rotation_difference_deg'], .1488, places=4)
        npy = load_calibration(Path(opencv['path']).with_name('eye_to_hand_20261002_143515_T_base_camera.npy'))
        np.testing.assert_allclose(npy['matrix'], opencv['matrix'])

    def test_metadata_missing_or_invalid_matrix_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'unknown_T_base_camera.npy'
            np.save(path, np.eye(4))
            with self.assertRaisesRegex(ValueError, 'metadata'):
                load_calibration(path)


if __name__ == '__main__':
    unittest.main()
