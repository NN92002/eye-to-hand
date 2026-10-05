"""Run the isolated LRBO point-cloud calibration frontend in Isaac Sim."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import tempfile
from concurrent.futures import ThreadPoolExecutor
from threading import Event
import time
from datetime import datetime
from pathlib import Path
import sys

import numpy as np

HERE = Path(__file__).resolve().parent
WORKSPACE_ROOT = HERE.parent.parent
LRBO_ROOT = Path(os.environ.get("LRBO_ROOT", Path.home() / "LRBO"))
LRBO_MODEL_PATH = LRBO_ROOT / "dataset" / "UR5e_base_m_x5.txt"
LRBO_CONFIG_PATH = LRBO_ROOT / "implementation" / "registration" / "test" / "indoor.yaml"
LRBO_CHECKPOINT_PATH = Path(
    os.environ.get(
        "LRBO_CHECKPOINT",
        LRBO_ROOT / "weights" / "registration" / "model_best_recall.pth",
    )
)
LRBO_PYTHON = Path(
    os.environ.get(
        "LRBO_PYTHON",
        Path.home() / "miniforge3" / "envs" / "lrbo-predator-gpu" / "bin" / "python",
    )
)
PREDATOR_ROOT = LRBO_ROOT / "third_party" / "OverlapPredator"
PREDATOR_WORKER = HERE / "predator_worker.py"
CAPTURE_DIR = HERE / "captures"
RESULT_DIR = HERE / "results"
BASE_ROI_PRIM_TOKENS = ("base_link_inertia", "shoulder_link")

sys.path.insert(0, str(HERE))
from lrbo_core import (  # noqa: E402
    depth_mask_to_camera_points,
    instance_ids_for_prim_tokens,
    LRBO_MODEL_SCALE,
    validate_rigid_transform,
    sample_triangle_surface,
)

SCENE_SCRIPT = WORKSPACE_ROOT / "isaac_sim" / "ur5e_camera_scene_charuco_calibration.py"
SCENE = None
CURRENT_CALIBRATOR = None
LRBO_UI = None


def _load_scene_module():
    spec = importlib.util.spec_from_file_location("ur5e_camera_scene_lrbo_host", SCENE_SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load Isaac Sim scene script: {SCENE_SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _rotation_error_deg(estimate, reference):
    delta = reference[:3, :3].T @ estimate[:3, :3]
    cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _rotation_mean(rotations):
    left, _, right = np.linalg.svd(np.mean(rotations, axis=0))
    correction = np.eye(3)
    correction[2, 2] = np.linalg.det(left @ right)
    return left @ correction @ right


def _timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


class LRBOEyeToHandCalibrator:
    """Capture base-link ROI scans and solve through an external LRBO runtime."""

    def __init__(self, stage, robot_base_path, tool_path, camera, dictionary, board):
        del tool_path, dictionary, board
        global CURRENT_CALIBRATOR
        CURRENT_CALIBRATOR = self
        self.stage = stage
        self.robot_base_path = robot_base_path
        self.camera = camera
        self.camera_matrix = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
        self.scans = []
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lrbo")
        self._solve_future = None
        self._closing = Event()
        self._generation = 0
        self.last_result = None
        self.last_status_text = "Waiting for camera segmentation data"
        self.latest_point_count = 0
        self.latest_prim_ids = np.empty((0,), dtype=np.uint32)
        camera.add_instance_id_segmentation_to_frame()
        self.publish_status(False)

    def publish_status(self, detected):
        translation_error = None
        rotation_error = None
        if self.last_result is not None:
            translation_error = self.last_result.get("translation_error_mm")
            rotation_error = self.last_result.get("rotation_error_deg")
        status = {
            "detected": bool(detected),
            "corners": int(self.latest_point_count),
            "samples": len(self.scans),
            "reproj_px": None,
            "text": self.last_status_text,
            "method": "LRBO PREDATOR + ICP",
            "translation_error_mm": translation_error,
            "rotation_error_deg": rotation_error,
            "method_diagnostics": {},
        }
        SCENE.push_calibration_status(status)
        if LRBO_UI is not None:
            LRBO_UI["scan"].text = (
                f"Base ROI: {self.latest_point_count:,} points | scans: {len(self.scans)}"
            )
            LRBO_UI["status"].text = self.last_status_text
            if self.last_result is None:
                LRBO_UI["result"].text = "LRBO result: not solved"
            elif translation_error is not None:
                LRBO_UI["result"].text = (
                    f"Sim GT error: {translation_error:.2f} mm | "
                    f"{rotation_error:.3f} deg"
                )
            else:
                LRBO_UI["result"].text = "LRBO result: solved; simulation GT unavailable"

    def update_detection(self, rgba, show_window=True):
        del rgba, show_window

    def _capture_base_roi(self):
        frame = self.camera.get_current_frame()
        segmentation_frame = frame.get("instance_id_segmentation")
        if not isinstance(segmentation_frame, dict):
            raise RuntimeError(
                "Instance-ID segmentation is not ready. Wait a few render frames and retry."
            )
        segmentation = segmentation_frame.get("data")
        segmentation_info = segmentation_frame.get("info", {})
        if segmentation is None:
            raise RuntimeError("Instance-ID segmentation returned no image data.")

        selected_ids = instance_ids_for_prim_tokens(
            segmentation_info,
            BASE_ROI_PRIM_TOKENS,
        )
        if selected_ids.size == 0:
            raise RuntimeError(
                "No segmentation IDs matched base_link_inertia or shoulder_link. "
                "Inspect the annotator idToLabels mapping and URDF prim paths."
            )

        # Read both annotators from the same camera acquisition snapshot.
        depth = frame.get("distance_to_image_plane")
        if depth is None:
            raise RuntimeError("Camera depth is not available yet.")
        self.camera_matrix = np.asarray(self.camera.get_intrinsics_matrix(), dtype=np.float64)
        points, pixel_count = depth_mask_to_camera_points(
            depth,
            segmentation,
            self.camera_matrix,
            selected_ids,
        )
        if len(points) < 100:
            raise RuntimeError(
                f"Base ROI contains only {len(points)} points; require at least 100."
            )
        self.latest_point_count = len(points)
        self.latest_prim_ids = selected_ids
        labels = segmentation_info.get("idToLabels", {})
        self.latest_prim_labels = [str(labels.get(str(int(i)), labels.get(int(i), "")))
                                   for i in selected_ids]
        return points, pixel_count

    def _capture_robot_model(self):
        """Sample the robot CAD at its current pose, without reading camera GT."""
        from pxr import Usd, UsdGeom
        root = self.stage.GetPrimAtPath(self.robot_base_path)
        if not root.IsValid():
            raise RuntimeError("Robot base prim is invalid")
        base_from_world = np.linalg.inv(SCENE.get_prim_world_matrix(
            self.stage, self.robot_base_path))
        triangles = []
        for prim in Usd.PrimRange(root):
            if not prim.IsA(UsdGeom.Mesh) or "/geometry/" not in str(prim.GetPath()).lower():
                continue
            selected = instance_ids_for_prim_tokens(
                {"idToLabels": {"1": str(prim.GetPath())}}, BASE_ROI_PRIM_TOKENS)
            if not len(selected):
                continue
            mesh = UsdGeom.Mesh(prim)
            vertices = np.asarray(mesh.GetPointsAttr().Get(), dtype=np.float64)
            if vertices.size == 0:
                continue
            transform = base_from_world @ SCENE.get_prim_world_matrix(self.stage, str(prim.GetPath()))
            vertices = vertices @ transform[:3, :3].T + transform[:3, 3]
            indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get(), dtype=int)
            offset = 0
            for count in mesh.GetFaceVertexCountsAttr().Get():
                face = indices[offset:offset + count]
                for k in range(1, count - 1):
                    triangles.append(vertices[[face[0], face[k], face[k + 1]]])
                offset += count
        if not triangles:
            raise RuntimeError("No base/shoulder visual mesh found; inspect robot Geometry prims")
        return sample_triangle_surface(triangles)

    def capture_sample(self):
        if self._solve_future is not None:
            self.last_status_text = "LRBO is running; wait before capturing"
            self.publish_status(bool(self.scans))
            return
        try:
            points, pixel_count = self._capture_base_roi()
            model_points = self._capture_robot_model()
            CAPTURE_DIR.mkdir(parents=True, exist_ok=True)
            capture_path = CAPTURE_DIR / f"lrbo_scan_{_timestamp()}.npz"
            ground_truth = np.full((4, 4), np.nan, dtype=np.float64)
            try:
                position, orientation = self.camera.get_world_pose(camera_axes="ros")
                T_world_camera = SCENE._matrix_from_rt(
                    SCENE.quaternion_wxyz_to_matrix(orientation), position
                )
                T_world_base = SCENE.get_prim_world_matrix(
                    self.stage,
                    self.robot_base_path,
                )
                ground_truth = np.linalg.inv(T_world_base) @ T_world_camera
            except Exception as error:
                SCENE.carb.log_warn(f"LRBO capture: simulation GT unavailable: {error!r}")

            np.savez_compressed(
                capture_path,
                points_camera_m=points,
                camera_matrix=self.camera_matrix,
                selected_instance_ids=self.latest_prim_ids,
                selected_pixel_count=np.asarray(pixel_count, dtype=np.int32),
                T_base_camera_GT=ground_truth,
                source_prim_tokens=np.asarray(BASE_ROI_PRIM_TOKENS),
                capture_version=np.asarray(3, dtype=np.int32),
                model_points_base_m=model_points,
                model_frame=np.asarray(self.robot_base_path),
                camera_frame=np.asarray("ROS optical: x right, y down, z forward"),
                robot_base_prim=np.asarray(self.robot_base_path),
                rendering_time=np.asarray(frame_time(self.camera)),
                selected_prim_labels=np.asarray(self.latest_prim_labels),
            )
            self.scans.append(capture_path)
            self.last_result = None
            self.last_status_text = f"Saved base ROI scan: {capture_path.name}"
            SCENE.carb.log_info(
                f"LRBO base ROI captured: points={len(points)}, "
                f"pixels={pixel_count}, file={capture_path}"
            )
            self.publish_status(True)
        except Exception as error:
            self.last_status_text = f"Capture failed: {error}"
            SCENE.carb.log_error(self.last_status_text)
            self.publish_status(False)

    def clear_samples(self):
        self._generation += 1
        self.scans.clear()
        self.last_result = None
        self.last_status_text = "Capture list cleared; saved scan files were kept"
        self.publish_status(False)

    def _run_predator_worker(self, scans):
        for label, path in (
            ("PREDATOR Python", LRBO_PYTHON),
            ("PREDATOR source", PREDATOR_ROOT),
            ("LRBO config", LRBO_CONFIG_PATH),
            ("PREDATOR checkpoint", LRBO_CHECKPOINT_PATH),
            ("PREDATOR worker", PREDATOR_WORKER),
        ):
            if not path.exists():
                raise RuntimeError(f"{label} not found: {path}")

        with tempfile.TemporaryDirectory(prefix="lrbo_predator_") as temp_dir:
            output_path = Path(temp_dir) / "registrations.npz"
            command = [
                str(LRBO_PYTHON),
                str(PREDATOR_WORKER),
                "--predator-root",
                str(PREDATOR_ROOT),
                "--config",
                str(LRBO_CONFIG_PATH),
                "--checkpoint",
                str(LRBO_CHECKPOINT_PATH),
                "--model",
                str(LRBO_MODEL_PATH),
                "--scans",
                *[str(path) for path in scans],
                "--device",
                os.environ.get("LRBO_DEVICE", "cuda"),
                "--output",
                str(output_path),
            ]
            environment = predator_environment()
            # File-backed logs avoid pipe deadlocks and unbounded memory use.
            with tempfile.TemporaryFile(mode="w+") as log:
                with subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                      text=True, env=environment, cwd=str(HERE)) as process:
                    deadline = time.monotonic() + 900
                    try:
                        while process.poll() is None:
                            if self._closing.wait(0.2):
                                raise RuntimeError("LRBO stopped because the scene closed")
                            if time.monotonic() >= deadline:
                                raise RuntimeError("PREDATOR worker exceeded the 900-second timeout")
                    finally:
                        if process.poll() is None:
                            process.terminate()
                            try:
                                process.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                process.kill()
                                process.wait()
                    if process.returncode != 0:
                        log.seek(0, os.SEEK_END)
                        log.seek(max(0, log.tell() - 6000))
                        raise RuntimeError(f"PREDATOR worker exited with code {process.returncode}: "
                                           + log.read())
            with np.load(output_path) as output:
                return {key: output[key].copy() for key in output.files}

    def solve(self):
        if not self.scans:
            self.last_status_text = "Capture at least one base ROI scan before solving"
            self.publish_status(False)
            return
        if not LRBO_MODEL_PATH.is_file():
            self.last_status_text = f"UR5e LRBO model cloud not found: {LRBO_MODEL_PATH}"
            self.publish_status(False)
            return

        if self._solve_future is not None:
            self.last_status_text = "LRBO is already running"
            self.publish_status(bool(self.scans))
            return
        self.last_result = None
        self.last_status_text = f"Running LRBO on {len(self.scans)} scan(s)..."
        self._solve_generation = self._generation
        self._solve_future = self._executor.submit(self._solve_scans, tuple(self.scans))
        self.publish_status(True)

    def _solve_scans(self, scans):
        try:
            for scan_path in scans:
                with np.load(scan_path) as scan:
                    if "model_points_base_m" not in scan:
                        raise ValueError("Legacy fixed-model capture: recapture with version 3 robot CAD model")
            worker_results = self._run_predator_worker(scans)
            results = []
            for index, scan_path in enumerate(scans):
                with np.load(scan_path) as scan:
                    ground_truth = scan["T_base_camera_GT"]
                result = {
                    "capture": str(scan_path),
                    "T_base_camera": worker_results["T_base_camera"][index],
                    "T_model_from_source_scaled": worker_results[
                        "T_model_from_source_scaled"
                    ][index],
                    "fitness": float(worker_results["fitness"][index]),
                    "inlier_rmse": float(worker_results["inlier_rmse"][index]),
                    "inlier_rmse_m": float(worker_results["inlier_rmse"][index]) / LRBO_MODEL_SCALE,
                    "source_points": int(worker_results["source_points"][index]),
                    "target_points": int(worker_results["target_points"][index]),
                }
                if np.all(np.isfinite(ground_truth)):
                    result["translation_error_mm"] = float(
                        np.linalg.norm(
                            result["T_base_camera"][:3, 3] - ground_truth[:3, 3]
                        )
                        * 1000.0
                    )
                    result["rotation_error_deg"] = _rotation_error_deg(
                        result["T_base_camera"],
                        ground_truth,
                    )
                results.append(result)

            transforms = [validate_rigid_transform(item["T_base_camera"]) for item in results]
            for i, first in enumerate(transforms):
                for second in transforms[i + 1:]:
                    if (np.linalg.norm(first[:3, 3] - second[:3, 3]) > 0.05
                            or _rotation_error_deg(first, second) > 5.0):
                        raise ValueError("Scans disagree (>50 mm / 5 deg); no average produced. "
                                         "Clear list, keep robot pose fixed and capture an unobstructed ROI.")
            estimate = np.eye(4, dtype=np.float64)
            estimate[:3, :3] = _rotation_mean(
                np.stack([transform[:3, :3] for transform in transforms])
            )
            estimate[:3, 3] = np.mean(
                np.stack([transform[:3, 3] for transform in transforms]),
                axis=0,
            )
            translation_spread_mm = max(float(np.linalg.norm(
                transform[:3, 3] - estimate[:3, 3]) * 1000.0) for transform in transforms)
            rotation_spread_deg = max(_rotation_error_deg(transform, estimate)
                                      for transform in transforms)
            gt_results = [item for item in results if "translation_error_mm" in item]
            translation_error = None
            rotation_error = None
            if gt_results:
                with np.load(gt_results[0]["capture"]) as capture:
                    gt = capture["T_base_camera_GT"].copy()
                for item in gt_results[1:]:
                    with np.load(item["capture"]) as capture:
                        if not np.allclose(gt, capture["T_base_camera_GT"], atol=1e-5):
                            raise ValueError("Camera/base moved between captures; clear the list and recapture.")
                translation_error = float(
                    np.linalg.norm(estimate[:3, 3] - gt[:3, 3]) * 1000.0
                )
                rotation_error = _rotation_error_deg(estimate, gt)

            result_summary = {
                "T_base_camera": estimate,
                "scans": results,
                "translation_error_mm": translation_error,
                "rotation_error_deg": rotation_error,
                "fitness_mean": float(np.mean([item["fitness"] for item in results])),
                "inlier_rmse_mean": float(
                    np.mean([item["inlier_rmse"] for item in results])
                ),
                "inlier_rmse_mean_m": float(np.mean([item["inlier_rmse_m"] for item in results])),
                "translation_spread_mm": translation_spread_mm,
                "rotation_spread_deg": rotation_spread_deg,
                "quality_passed": all(0.3 <= item["fitness"] <= 1.0 and 0.0 <= item["inlier_rmse_m"] <= 0.01 for item in results)
                and translation_spread_mm <= 50.0 and rotation_spread_deg <= 5.0
                and (translation_error is None or (translation_error <= 50.0 and rotation_error <= 5.0)),
            }
            return result_summary
        except Exception as error:
            raise RuntimeError(f"LRBO solve failed: {error}") from error

    def _poll_solve(self):
        if self._solve_future is None or not self._solve_future.done():
            return
        future, self._solve_future = self._solve_future, None
        if self._solve_generation != self._generation:
            self.last_status_text = "Discarded result because capture list was cleared"
            self.publish_status(bool(self.scans))
            return
        try:
            self.last_result = future.result()
            if self.last_result["quality_passed"]:
                self.last_status_text = f"LRBO solved from {len(self.last_result['scans'])} scan(s)"
            else:
                self.last_status_text = "LOW ACCURACY: inspect ROI, model frame and registration quality"
            SCENE.carb.log_info("LRBO eye-to-hand result:\n" + np.array2string(
                self.last_result["T_base_camera"], precision=7, suppress_small=True))
        except Exception as error:
            self.last_status_text = str(error)
            SCENE.carb.log_error(self.last_status_text)
        self.publish_status(bool(self.scans))

    def close(self):
        self._closing.set()
        self._executor.shutdown(wait=True, cancel_futures=True)

    def save(self):
        if self.last_result is None:
            self.last_status_text = "Solve successfully before saving an LRBO result"
            self.publish_status(bool(self.scans))
            return
        if not self.last_result.get("quality_passed", True):
            self.last_status_text = (
                "SAVE rejected: registration quality or simulation GT check failed; "
                "inspect ROI/frame alignment first"
            )
            self.publish_status(bool(self.scans))
            return
        RESULT_DIR.mkdir(parents=True, exist_ok=True)
        stem = RESULT_DIR / f"lrbo_eye_to_hand_{_timestamp()}"
        np.save(str(stem) + "_T_base_camera.npy", self.last_result["T_base_camera"])
        metadata = {
            "method": "LRBO PREDATOR + Open3D ICP",
            "parent_frame": self.robot_base_path,
            "model_source": "robot visual CAD at each capture pose",
            "evaluation_base_prim": self.robot_base_path,
            "model_path": str(LRBO_MODEL_PATH),
            "registration_scale": LRBO_MODEL_SCALE,
            "translation_spread_mm": self.last_result["translation_spread_mm"],
            "rotation_spread_deg": self.last_result["rotation_spread_deg"],
            "child_frame": "camera_color_optical_frame",
            "convention": "T_A_B maps coordinates from frame B to frame A",
            "T_base_camera": self.last_result["T_base_camera"].tolist(),
            "translation_error_mm": self.last_result["translation_error_mm"],
            "rotation_error_deg": self.last_result["rotation_error_deg"],
            "fitness_mean": self.last_result["fitness_mean"],
            "inlier_rmse_mean": self.last_result["inlier_rmse_mean"],
            "inlier_rmse_mean_m": self.last_result["inlier_rmse_mean_m"],
            "scans": [
                {
                    key: value.tolist() if isinstance(value, np.ndarray) else value
                    for key, value in result.items()
                }
                for result in self.last_result["scans"]
            ],
        }
        Path(str(stem) + ".json").write_text(json.dumps(metadata, indent=2) + "\n")
        self.last_status_text = f"Saved LRBO result: {stem.name}"
        self.publish_status(True)

    def process_commands(self):
        self._poll_solve()
        actions = {
            "capture": self.capture_sample,
            "solve": self.solve,
            "clear": self.clear_samples,
            "save": self.save,
        }
        while True:
            try:
                command = SCENE.CALIBRATION_COMMAND_QUEUE.get_nowait()
            except SCENE.queue.Empty:
                break
            if command in actions:
                actions[command]()


def predator_environment(parent=None):
    """Keep Isaac's Python and native libraries out of the separate runtime."""
    environment = dict(os.environ if parent is None else parent)
    for key in ("PYTHONHOME", "PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE",
                "PYTHONEXECUTABLE", "PYTHONEXE", "VIRTUAL_ENV", "CONDA_PREFIX",
                "CONDA_DEFAULT_ENV", "LD_PRELOAD"):
        environment.pop(key, None)
    environment["PYTHONNOUSERSITE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join((str(HERE), str(PREDATOR_ROOT)))
    isaac_root = environment.get("ISAAC_PATH", "/home/neaf6000pro/isaacsim")
    def outside_isaac(value):
        return bool(value) and not (
            value == isaac_root or value.startswith(isaac_root.rstrip("/") + "/"))
    for key in ("LD_LIBRARY_PATH", "PATH"):
        entries = [entry for entry in environment.get(key, "").split(os.pathsep)
                   if outside_isaac(entry)]
        if key == "PATH":
            entries.insert(0, str(LRBO_PYTHON.parent))
        if entries:
            environment[key] = os.pathsep.join(entries)
        else:
            environment.pop(key, None)
    return environment


def frame_time(camera):
    return float(camera.get_current_frame().get("rendering_time", 0.0))


def _start_lrbo_gui(scene):
    global LRBO_UI
    original_label = scene.ui.Label
    original_button = scene.ui.Button

    def hide_label(text):
        value = str(text)
        return (
            "ChArUco" in value
            or value.startswith("Eye-to-Hand ChArUco")
            or value.startswith("Samples:")
            or value.startswith("Isaac Sim GT error:")
            or value.startswith("Solver:")
            or value.startswith("PARK:")
            or value.startswith("ANDREFF:")
        )

    def filtered_label(*args, **kwargs):
        widget = original_label(*args, **kwargs)
        text = args[0] if args else kwargs.get("text", "")
        if hide_label(text):
            widget.visible = False
        return widget

    def filtered_button(text, *args, **kwargs):
        widget = original_button(text, *args, **kwargs)
        if str(text) in {"CAPTURE", "SOLVE", "CLEAR", "SAVE"}:
            widget.visible = False
        return widget

    scene.ui.Label = filtered_label
    scene.ui.Button = filtered_button
    try:
        joint_gui = scene._original_start_joint_control_gui()
    finally:
        scene.ui.Label = original_label
        scene.ui.Button = original_button

    window = scene.ui.Window("LRBO Eye-to-Hand", width=440, height=170)
    with window.frame:
        with scene.ui.VStack(spacing=6, height=0):
            scene.ui.Label("UR5e base registration", height=24)
            scan_label = scene.ui.Label("Base ROI: waiting | scans: 0", height=22)
            status_label = scene.ui.Label("Waiting for camera segmentation", height=22)
            result_label = scene.ui.Label("LRBO result: not solved", height=22)
            with scene.ui.HStack(height=34, spacing=8):
                scene.ui.Button(
                    "CAPTURE BASE ROI",
                    clicked_fn=lambda: scene.enqueue_calibration_command("capture"),
                )
                scene.ui.Button(
                    "RUN LRBO",
                    clicked_fn=lambda: scene.enqueue_calibration_command("solve"),
                )
                scene.ui.Button(
                    "CLEAR LIST",
                    clicked_fn=lambda: scene.enqueue_calibration_command("clear"),
                )
                scene.ui.Button(
                    "SAVE RESULT",
                    clicked_fn=lambda: scene.enqueue_calibration_command("save"),
                )
    LRBO_UI = {"window": window, "scan": scan_label, "status": status_label, "result": result_label}
    if CURRENT_CALIBRATOR is not None:
        CURRENT_CALIBRATOR.publish_status(False)
    return joint_gui


def main():
    global SCENE
    SCENE = _load_scene_module()
    SCENE.create_charuco_board_definition = lambda: (None, None)
    SCENE.generate_charuco_texture = lambda _board: None
    SCENE.create_charuco_board_prim = lambda _stage, _tool_path: "LRBO uses no ChArUco target"
    SCENE.initialize_depth_window = lambda: None
    SCENE.show_depth = SCENE.make_depth_heatmap
    SCENE.EyeToHandCalibrator = LRBOEyeToHandCalibrator
    SCENE._original_start_joint_control_gui = SCENE.start_joint_control_gui
    SCENE.start_joint_control_gui = lambda: _start_lrbo_gui(SCENE)
    try:
        SCENE.main()
    finally:
        if CURRENT_CALIBRATOR is not None:
            CURRENT_CALIBRATOR.close()


if __name__ == "__main__":
    main()
