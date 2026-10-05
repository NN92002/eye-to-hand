"""Compare exported OpenCV and LRBO camera extrinsics without Isaac imports."""
from __future__ import annotations

import ast
import json
from pathlib import Path

import numpy as np

from lrbo_core import validate_rigid_transform

CONVENTION = "T_A_B maps coordinates from frame B to frame A"


def rotation_difference_deg(first, second):
    relative = first[:3, :3].T @ second[:3, :3]
    cosine = np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def _read_exported_yaml(path):
    """Read only the known scalar and matrix fields in our scene's YAML export.

    No YAML dependency is needed in Isaac; arbitrary YAML features are rejected.
    """
    metadata = {}
    rows = []
    in_matrix = False
    for line in path.read_text().splitlines():
        if line == "T_base_camera:":
            in_matrix = True
            continue
        if in_matrix:
            if line.startswith("  - ["):
                rows.append(ast.literal_eval(line.strip()[2:]))
                continue
            in_matrix = False
        if line and not line[0].isspace() and ":" in line:
            key, value = line.split(":", 1)
            metadata[key] = value.strip().strip("'\"")
    metadata["T_base_camera"] = rows
    return metadata


def load_calibration(path):
    """Load export with frame metadata and optional capture-ground-truth evidence."""
    path = Path(path).expanduser().resolve()
    if path.suffix == ".npy":
        stem = str(path).removesuffix("_T_base_camera.npy")
        if stem == str(path):
            raise ValueError("Choose an exported *_T_base_camera.npy or its YAML/JSON")
        companions = [Path(stem + suffix) for suffix in (".yaml", ".json")]
        companion = next((p for p in companions if p.is_file()), None)
        if companion is None:
            raise ValueError("Matrix has no companion frame metadata; choose YAML/JSON export")
        record = load_calibration(companion)
        matrix = validate_rigid_transform(np.load(path, allow_pickle=False))
        if not np.allclose(matrix, record["matrix"], atol=1e-7):
            raise ValueError("NPY matrix disagrees with its metadata export")
        return record
    if path.suffix == ".json":
        metadata = json.loads(path.read_text())
    elif path.suffix in (".yaml", ".yml"):
        metadata = _read_exported_yaml(path)
    else:
        raise ValueError("Choose an exported YAML, JSON or *_T_base_camera.npy")
    if metadata.get("convention") != CONVENTION:
        raise ValueError("Missing/unsupported transform direction convention")
    if metadata.get("child_frame") != "camera_color_optical_frame":
        raise ValueError("Expected ROS camera_color_optical_frame")
    if not metadata.get("parent_frame"):
        raise ValueError("Missing parent frame")
    matrix = validate_rigid_transform(np.asarray(metadata["T_base_camera"], dtype=float))
    ground_truths = []
    gt_file = path.with_name(path.stem + "_T_base_camera_GT.npy")
    if gt_file.is_file():
        ground_truths.append(validate_rigid_transform(np.load(gt_file, allow_pickle=False)))
    # LRBO JSON refers to captures containing the actual base/camera GT.
    for scan in metadata.get("scans", []):
        capture = Path(scan["capture"])
        if capture.is_file():
            with np.load(capture, allow_pickle=False) as data:
                gt = data.get("T_base_camera_GT")
                if gt is not None and np.isfinite(gt).all():
                    ground_truths.append(validate_rigid_transform(gt))
    if ground_truths and any(not np.allclose(ground_truths[0], gt, atol=1e-5, rtol=0)
                             for gt in ground_truths[1:]):
        raise ValueError("Capture ground truths disagree: camera/base moved between samples")
    return {"path": str(path), "matrix": matrix, "metadata": metadata,
            "ground_truth": ground_truths[0] if ground_truths else None}


def compare_calibrations(opencv, lrbo):
    """Require common frames/poses, return difference and optional GT errors."""
    first_meta, second_meta = opencv["metadata"], lrbo["metadata"]
    if first_meta.get("child_frame") != second_meta.get("child_frame"):
        raise ValueError("Camera frame mismatch")
    first_gt, second_gt = opencv["ground_truth"], lrbo["ground_truth"]
    if first_gt is not None and second_gt is not None:
        if not np.allclose(first_gt, second_gt, atol=1e-5, rtol=0):
            raise ValueError("Saved camera/base poses differ; cannot compare as one calibration setup")
        frame_check = "Common base/camera pose verified from capture GT (GT is not used to fit)"
    else:
        if first_meta["parent_frame"] != second_meta["parent_frame"]:
            raise ValueError("Parent frames differ and GT evidence is unavailable; cannot assume base alias")
        frame_check = "Frame names match; historical pose could not be verified (GT missing)"
    a, b = opencv["matrix"], lrbo["matrix"]
    delta_mm = (b[:3, 3] - a[:3, 3]) * 1000.0
    result = {
        "opencv_file": opencv["path"], "lrbo_file": lrbo["path"],
        "opencv_method": first_meta.get("method"), "lrbo_method": second_meta.get("method"),
        "frame_check": frame_check, "convention": CONVENTION,
        "translation_delta_lrbo_minus_opencv_mm": delta_mm.tolist(),
        "translation_difference_mm": float(np.linalg.norm(delta_mm)),
        "rotation_difference_deg": rotation_difference_deg(a, b),
        "T_base_camera_opencv": a.tolist(), "T_base_camera_lrbo": b.tolist(),
        "T_relative_inv_opencv_times_lrbo": (np.linalg.inv(a) @ b).tolist(),
    }
    if first_gt is not None and second_gt is not None:
        for name, matrix in (("opencv", a), ("lrbo", b)):
            result[f"{name}_gt_translation_error_mm"] = float(
                np.linalg.norm(matrix[:3, 3] - first_gt[:3, 3]) * 1000)
            result[f"{name}_gt_rotation_error_deg"] = rotation_difference_deg(first_gt, matrix)
    return result


def comparison_text(result):
    delta = result["translation_delta_lrbo_minus_opencv_mm"]
    lines = [f"OpenCV: {Path(result['opencv_file']).name}",
             f"LRBO: {Path(result['lrbo_file']).name}", result["frame_check"],
             f"Translation difference: {result['translation_difference_mm']:.3f} mm",
             f"Delta xyz (LRBO - OpenCV): {delta[0]:+.3f}, {delta[1]:+.3f}, {delta[2]:+.3f} mm",
             f"Rotation difference: {result['rotation_difference_deg']:.4f} deg"]
    for name in ("opencv", "lrbo"):
        if f"{name}_gt_translation_error_mm" in result:
            lines.append(f"{name.upper()} vs sim GT: "
                         f"{result[f'{name}_gt_translation_error_mm']:.3f} mm / "
                         f"{result[f'{name}_gt_rotation_error_deg']:.4f} deg")
    lines.append("Difference alone does not establish accuracy.")
    return "\n".join(lines)
