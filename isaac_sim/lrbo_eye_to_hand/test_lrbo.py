"""Offline regression checks; no Isaac Sim installation needed."""
import importlib.util
from pathlib import Path
import sys
import unittest
from concurrent.futures import Future
from types import SimpleNamespace

import numpy as np

from lrbo_core import (
    depth_mask_to_camera_points,
    instance_ids_for_prim_tokens,
    lrbo_transform_to_base_camera,
    run_lrbo_registration,
    sample_triangle_surface,
)


class GeometryTests(unittest.TestCase):
    def test_cad_sampling_preserves_surface(self):
        triangle = np.array([[[0, 0, 2], [1, 0, 2], [0, 1, 2]]])
        points = sample_triangle_surface(triangle, count=1000)
        np.testing.assert_allclose(points[:, 2], 2)
        self.assertTrue(np.all(points[:, :2] >= 0))
        self.assertTrue(np.all(points[:, :2].sum(1) <= 1.000001))
        np.testing.assert_array_equal(points, sample_triangle_surface(triangle, count=1000))

    def test_camera_to_base_direction_and_scale(self):
        angle = 0.7
        rotation = np.array([[np.cos(angle), -np.sin(angle), 0],
                             [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
        expected = np.eye(4)
        expected[:3, :3] = rotation
        expected[:3, 3] = [0.8, -0.3, 0.9]
        registered = expected.copy()
        registered[:3, 3] *= 5
        np.testing.assert_allclose(lrbo_transform_to_base_camera(registered), expected)

    def test_reject_nonrigid(self):
        for invalid in [np.diag([1, 1, -1, 1]), np.diag([2, 2, 2, 1])]:
            with self.assertRaises(ValueError):
                lrbo_transform_to_base_camera(invalid)

    def test_roi_excludes_nested_arm(self):
        info = {"idToLabels": {
            "1": "/World/ur5e/base_link_inertia/visual/mesh",
            "2": {"primPath": "/World/ur5e/shoulder_link/visual/mesh"},
            "3": "/World/ur5e/shoulder_link/upper_arm_link/visual/mesh",
            "4": "/World/ur5e/base_link_inertia/shoulder_link/forearm_link/mesh",
            "5": "/World/ur5e/not_shoulder_link/mesh",
        }}
        np.testing.assert_array_equal(instance_ids_for_prim_tokens(
            info, ("base_link_inertia", "shoulder_link")), [1, 2])

    def test_backprojection_and_invalid_depth(self):
        depth = np.array([[1, np.nan, 2], [0, 3, np.inf]])
        ids = np.array([[7, 7, 8], [7, 7, 7]])
        K = np.array([[2, 0, 1], [0, 2, 0], [0, 0, 1]])
        points, count = depth_mask_to_camera_points(depth, ids, K, [7])
        self.assertEqual(count, 2)
        np.testing.assert_allclose(points, [[-0.5, 0, 1], [0, 1.5, 3]])
        transposed, _ = depth_mask_to_camera_points(depth, ids.T[..., None], K, [7])
        np.testing.assert_allclose(transposed, points)

    def test_end_to_end_synthetic_registration(self):
        # Known initialization checks the real Open3D refinement and frame conversion.
        rng = np.random.default_rng(4)
        target_m = rng.uniform(-0.2, 0.2, (1500, 3)).astype(np.float32)
        theta = 0.4
        R = np.array([[np.cos(theta), -np.sin(theta), 0],
                      [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        t = np.array([0.8, -0.3, 0.9])
        source = (target_m - t) @ R
        initial = np.eye(4)
        initial[:3, :3] = R
        initial[:3, 3] = t * 5
        result = run_lrbo_registration(source, target_m * 5, lambda *_: np.eye(4),
                                       initial_candidates=(initial,))
        np.testing.assert_allclose(result["T_base_camera"][:3, :3], R, atol=0.002)
        np.testing.assert_allclose(result["T_base_camera"][:3, 3], t, atol=0.001)
        self.assertGreater(result["fitness"], 0.95)


class LifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).with_name("run_lrbo_isaac.py")
        spec = importlib.util.spec_from_file_location("lrbo_frontend_test", path)
        cls.module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = cls.module
        spec.loader.exec_module(cls.module)

    def test_clear_discards_pending_result(self):
        obj = self.module.LRBOEyeToHandCalibrator.__new__(self.module.LRBOEyeToHandCalibrator)
        obj.scans = ["scan"]
        obj._generation = obj._solve_generation = 0
        obj._solve_future = Future()
        obj.last_result = None
        obj.publish_status = lambda *_: None
        obj.clear_samples()
        obj._solve_future.set_result({"quality_passed": True})
        obj._poll_solve()
        self.assertIsNone(obj.last_result)
        self.assertIsNone(obj._solve_future)

    def test_worker_isolated_from_isaac_python(self):
        import os
        import json
        import subprocess
        contaminated = dict(os.environ)
        root = "/home/neaf6000pro/isaacsim"
        contaminated.update({
            "ISAAC_PATH": root,
            "PYTHONPATH": root + "/kit/python/lib/python3.12",
            "PYTHONHOME": root + "/kit/python",
            "LD_PRELOAD": root + "/kit/libcarb.so",
            "LD_LIBRARY_PATH": root + "/kit:/usr/lib",
        })
        environment = self.module.predator_environment(contaminated)
        completed = subprocess.run(
            [str(self.module.LRBO_PYTHON), "-c",
             "import argparse,re,json,sys,torch,open3d; "
             "print(json.dumps({'version':sys.version_info[:2],'re':re.__file__}))"],
            env=environment, check=True, capture_output=True, text=True, timeout=30)
        info = json.loads(completed.stdout)
        self.assertEqual(info["version"], [3, 10])
        self.assertNotIn(root, info["re"])
        self.assertNotIn("LD_PRELOAD", environment)
        self.assertEqual(environment["LD_LIBRARY_PATH"], "/usr/lib")

    def test_capture_uses_acquired_depth_snapshot(self):
        obj = self.module.LRBOEyeToHandCalibrator.__new__(self.module.LRBOEyeToHandCalibrator)
        frame = {
            "distance_to_image_plane": np.ones((12, 12)),
            "instance_id_segmentation": {
                "data": np.ones((12, 12), dtype=np.uint32),
                "info": {"idToLabels": {"1": "/World/robot/shoulder_link/mesh"}},
            },
        }
        obj.camera = SimpleNamespace(
            get_current_frame=lambda: frame,
            get_intrinsics_matrix=lambda: np.array([[10, 0, 6], [0, 10, 6], [0, 0, 1]]),
        )
        points, count = obj._capture_base_roi()
        self.assertEqual(count, 144)
        self.assertEqual(points.shape, (144, 3))
        self.assertEqual(len(obj.latest_prim_labels), 1)

    def test_solve_rejects_zero_overlap_without_gt(self):
        import tempfile
        obj = self.module.LRBOEyeToHandCalibrator.__new__(self.module.LRBOEyeToHandCalibrator)
        obj._run_predator_worker = lambda _: {
            "T_base_camera": np.eye(4)[None],
            "T_model_from_source_scaled": np.eye(4)[None],
            "fitness": np.array([0.0]), "inlier_rmse": np.array([0.0]),
            "source_points": np.array([200]), "target_points": np.array([200]),
        }
        with tempfile.TemporaryDirectory() as directory:
            capture = Path(directory) / "capture.npz"
            np.savez(capture, T_base_camera_GT=np.full((4, 4), np.nan),
                     model_points_base_m=np.zeros((200, 3)))
            result = obj._solve_scans((capture,))
        self.assertFalse(result["quality_passed"])

    def test_score_sampling_handles_sparse_scores(self):
        spec = importlib.util.spec_from_file_location("worker_test", Path(__file__).with_name("predator_worker.py"))
        worker = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(worker)
        points = np.arange(60).reshape(20, 3)
        scores = np.zeros(20)
        scores[0] = 1
        sampled, _ = worker._sample_by_score(points, points, scores, 10)
        self.assertEqual(len(np.unique(sampled, axis=0)), 10)


if __name__ == "__main__":
    unittest.main()
