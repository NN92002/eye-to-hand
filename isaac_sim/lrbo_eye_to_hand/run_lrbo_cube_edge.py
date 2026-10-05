"""Existing LRBO scene with a random green cube and RGB edge overlay and aligned depth sampling.

The host scene/main loop and manual joint controls are reused without editing
them. Edge detection is button-triggered and frozen until retriggered. Only saved
OpenCV/LRBO extrinsics are loaded and compared; no calibration runs here.
"""
from __future__ import annotations

import time
import os
import sys
import json
from pathlib import Path

import cv2
import numpy as np

import run_lrbo_isaac as lrbo
from calibration_comparison import load_calibration, compare_calibrations, comparison_text
from edge_tip_path import (URDFModel, build_edge_plan, draw_plan, SlowEdgeExecutor,
                           transform_points, unit)

from edge_path_evaluation import evaluate_path, evaluation_text, draw_ideal_reference
from edge_surface_depth import fit_edge_surface_depth

CURRENT_DETECTOR = None
COMPARISON_DIR = Path(__file__).resolve().parent / "comparisons"


def default_calibration_file(environment_key, directory, pattern):
    override = os.environ.get(environment_key)
    if override:
        return override
    files = sorted(Path(directory).glob(pattern))
    return str(files[-1]) if files else ""

# OpenCV HSV: H = 0..179, S/V = 0..255. Tune these first using Green Mask.
GREEN_HSV_LOWER = (40, 80, 35)
GREEN_HSV_UPPER = (85, 255, 255)
MORPH_KERNEL_SIZE = 3  # positive odd integer
MORPH_OPEN_ITERATIONS = 1
MORPH_CLOSE_ITERATIONS = 2
MIN_CONTOUR_AREA_PX = 80.0
MIN_EDGE_LENGTH_PX = 12.0
CANNY_LOW = 25
CANNY_HIGH = 70
HOUGH_THRESHOLD = 12
HOUGH_MAX_GAP_PX = 3
MIN_LINE_SUPPORT = 0.70
EDGE_SIDE_OFFSET_PX = 3
MIN_VALID_SIDE_RATIO = 0.65
MAX_INTERNAL_DEPTH_JUMP_M = 0.04
OCCLUDER_DEPTH_MARGIN_M = 0.015
DETECTION_HZ = 15.0
CUBE_SIZE_M = 0.12
TABLE_INSET_M = 0.005  # small clearance in addition to the cube half-width
# Nearby surface depth is estimated inside the selected cube mask, not at the
# ambiguous silhouette pixel itself. All distances are camera optical Z in m.
DEPTH_INSET_PX = 2
DEPTH_SEARCH_RADIUS_PX = 6
DEPTH_PATCH_RADIUS_PX = 2
EDGE_DEPTH_SAMPLE_COUNT = 41
RGB_WINDOW_NAME = "D435 RGB - Cube Edge Detection"
MASK_WINDOW_NAME = "Green Mask"

# Execution requires the EXECUTE PATH button after a successful preview.
EXECUTE_ROBOT = True
TOOL_TILT_DEG = 45.0
EDGE_CLEARANCE = 0.020  # tip path above the detected edge along table normal (m)
PRE_START_CLEARANCE = 0.080
WAYPOINT_SPACING = 0.002
TCP_TO_TIP_OFFSET = None  # auto from exact URDF, closed 2F-140 (~[0, 0, .23316] m)
TCP_TIP_AXIS = None  # auto from tool0 -> gripper-base mounting transform
TABLE_NORMAL = np.array([0., 0., 1.])  # calibration base frame; current base is yaw-only
TCP_WORKSPACE_MARGIN = 0.30  # TCP can be outside tabletop XY while tip is over it
TIP_SPEED_M_S = 0.030
JOINT_SPEED_RAD_S = 0.30
JOINT_ACCEL_RAD_S2 = 1.80
MIN_WAYPOINT_DURATION_S = 0.100
MAX_JOINT_WAYPOINT_JUMP = np.radians(8)
MAX_CARTESIAN_WAYPOINT_JUMP = 0.010
MAX_TRACKING_ERROR_RAD = np.radians(10)
MAX_TIP_TRACKING_ERROR_M = 0.010
PATH_ARM_STIFFNESS = 10000.0
PATH_ARM_DAMPING = 1000.0
DEPTH_LINE_CONSISTENCY_M = 0.025


def spawn_random_green_cube(world, stage):
    """Use the host's FixedCuboid API; cube stays fixed on the tabletop."""
    scene = lrbo.SCENE
    half = CUBE_SIZE_M / 2.0
    center_xy = np.asarray(scene.WORKSPACE_CENTER[:2], dtype=np.float64)
    extent_xy = np.array([scene.TABLE_WIDTH, scene.TABLE_DEPTH]) / 2.0
    rng = np.random.default_rng()
    yaw = rng.uniform(0.0, 2.0 * np.pi)
    # Rotated square's XY half-extent: prevents corners protruding off table.
    rotated_half = half * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
    allowed = extent_xy - rotated_half - TABLE_INSET_M
    if np.any(allowed <= 0):
        raise ValueError("Cube does not fit inside the workspace")
    xy = rng.uniform(center_xy - allowed, center_xy + allowed)
    tabletop_z = float(scene.TABLE_HEIGHT + scene.TABLE_THICKNESS)
    position = np.array([xy[0], xy[1], tabletop_z + half], dtype=np.float64)
    path = "/GreenCube"
    if stage.GetPrimAtPath(path).IsValid():
        stage.RemovePrim(path)
    cube = scene.FixedCuboid(
        prim_path=path,
        name="green_cube",
        position=position,
        orientation=np.array([np.cos(yaw / 2.0), 0.0, 0.0, np.sin(yaw / 2.0)]),
        size=1.0,
        scale=np.full(3, CUBE_SIZE_M),
        color=np.array([0.0, 1.0, 0.0]),
    )
    world.scene.add(cube)
    print(f"Green cube world position (m): {tuple(position)}; size: {CUBE_SIZE_M} m; yaw: {np.degrees(yaw):.1f} deg",
          flush=True)
    return cube


def detect_green_cube(rgb_frame, color_order="RGB"):
    """Return cleaned mask, largest valid contour and its pixel centroid."""
    conversion = {"RGB": cv2.COLOR_RGB2HSV, "BGR": cv2.COLOR_BGR2HSV}
    if color_order not in conversion:
        raise ValueError("color_order must be RGB or BGR")
    hsv = cv2.cvtColor(rgb_frame, conversion[color_order])
    mask = cv2.inRange(hsv, np.array(GREEN_HSV_LOWER, dtype=np.uint8),
                       np.array(GREEN_HSV_UPPER, dtype=np.uint8))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,
                                      (MORPH_KERNEL_SIZE, MORPH_KERNEL_SIZE))
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel,
                            iterations=MORPH_OPEN_ITERATIONS)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel,
                            iterations=MORPH_CLOSE_ITERATIONS)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = [c for c in contours if cv2.contourArea(c) >= MIN_CONTOUR_AREA_PX]
    if not valid:
        return mask, None, None
    contour = max(valid, key=cv2.contourArea)
    moments = cv2.moments(contour)
    if moments["m00"] <= 0:
        return mask, None, None
    center = (int(round(moments["m10"] / moments["m00"])),
              int(round(moments["m01"] / moments["m00"])))
    return mask, contour, center


def extract_edges(rgb_frame, mask, contour, depth=None):
    """Use real RGB gradients for line candidates, never polygon mask sides.

    Internal lines need green surface on both sides. Silhouette candidates also
    need depth proving the non-green side is behind the cube; foreground robot
    occlusion boundaries are rejected. This is evidence filtering, not a proof
    that a line is a physical cube ridge (shadows can still generate RGB edges).
    """
    if contour is None:
        return []
    selected = np.zeros_like(mask)
    cv2.drawContours(selected, [contour], -1, 255, cv2.FILLED)
    selected = cv2.bitwise_and(selected, mask)
    x, y, width, height = cv2.boundingRect(contour)
    pad = EDGE_SIDE_OFFSET_PX + 3
    x0, y0 = max(0, x - pad), max(0, y - pad)
    x1, y1 = min(mask.shape[1], x + width + pad), min(mask.shape[0], y + height + pad)
    gray = cv2.cvtColor(rgb_frame[y0:y1, x0:x1], cv2.COLOR_RGB2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    # Run Canny on RGB-derived gray before applying ROI mask. Masking gray first
    # would synthesize a false edge along the mask boundary.
    canny = cv2.Canny(gray, CANNY_LOW, CANNY_HIGH)
    roi_mask = cv2.dilate(selected[y0:y1, x0:x1], np.ones((3, 3), np.uint8))
    canny[roi_mask == 0] = 0
    lines = cv2.HoughLinesP(canny, 1, np.pi / 180, HOUGH_THRESHOLD,
                            minLineLength=MIN_EDGE_LENGTH_PX, maxLineGap=HOUGH_MAX_GAP_PX)
    if lines is None:
        return []
    supported = cv2.dilate(canny, np.ones((3, 3), np.uint8)) > 0
    depth_map = None
    if depth is not None:
        candidate = np.asarray(depth)
        if candidate.ndim == 3 and candidate.shape[-1] == 1:
            candidate = candidate[..., 0]
        if candidate.shape == mask.shape:
            depth_map = candidate
    edges = []
    for line in lines[:, 0]:
        first, second = line[:2] + [x0, y0], line[2:] + [x0, y0]
        delta = second.astype(float) - first
        length = float(np.linalg.norm(delta))
        if length < MIN_EDGE_LENGTH_PX:
            continue
        points = np.rint(first + np.linspace(0, 1, int(np.ceil(length)) + 1)[:, None] * delta).astype(int)
        support = float(np.mean(supported[points[:, 1] - y0, points[:, 0] - x0]))
        if support < MIN_LINE_SUPPORT:
            continue
        normal = np.array([-delta[1], delta[0]]) / length
        # Avoid corners where perpendicular probes can leave both cube faces.
        middle = first + np.linspace(.15, .85, 21)[:, None] * delta
        sides = [np.rint(middle + sign * EDGE_SIDE_OFFSET_PX * normal).astype(int)
                 for sign in (-1, 1)]
        inside = [((q[:, 0] >= 0) & (q[:, 0] < mask.shape[1]) &
                   (q[:, 1] >= 0) & (q[:, 1] < mask.shape[0])) for q in sides]
        usable = inside[0] & inside[1]
        if np.mean(usable) < MIN_VALID_SIDE_RATIO:
            continue
        left, right = (q[usable] for q in sides)
        green_left = selected[left[:, 1], left[:, 0]] > 0
        green_right = selected[right[:, 1], right[:, 0]] > 0
        internal = green_left & green_right
        silhouette = green_left ^ green_right
        accepted = internal.copy()
        if depth_map is not None:
            zl, zr = depth_map[left[:, 1], left[:, 0]], depth_map[right[:, 1], right[:, 0]]
            finite = np.isfinite(zl) & np.isfinite(zr) & (zl > 0) & (zr > 0)
            accepted &= finite & (np.abs(zl - zr) <= MAX_INTERNAL_DEPTH_JUMP_M)
            cube_z = np.where(green_left, zl, zr)
            background_z = np.where(green_left, zr, zl)
            accepted |= silhouette & finite & (background_z > cube_z + OCCLUDER_DEPTH_MARGIN_M)
        if np.mean(accepted) < MIN_VALID_SIDE_RATIO:
            continue
        kind = "internal RGB line" if np.mean(internal) >= MIN_VALID_SIDE_RATIO else "depth-checked silhouette"
        p1, p2 = sorted((tuple(map(int, first)), tuple(map(int, second))))
        # Hough often returns several nearly identical parallel segments.
        duplicate = any(np.linalg.norm(np.array(e['p1']) - p1) < 5 and
                        np.linalg.norm(np.array(e['p2']) - p2) < 5 for e in edges)
        if not duplicate:
            edges.append({"p1": p1, "p2": p2, "length_px": length,
                          "kind": kind, "support": support})
    return edges


def select_target_edge(edges):
    """Select the longest candidate in pixel space; deterministic tie-breaking."""
    return max(edges, key=lambda e: (e["length_px"], e["p1"], e["p2"]), default=None)


def sample_edge_depth(depth, mask, contour, target_edge):
    """Sample nearby cube surface Z, preserving exact endpoint pixel depths too.

    Reject mismatched image sizes rather than silently resize a depth map.
    Samples remain in pixel/depth space; no back projection or calibration.
    """
    result = {"status": "depth unavailable", "samples": [], "p1_raw_m": None,
              "p2_raw_m": None, "p1_surface_m": None, "p2_surface_m": None,
              "median_m": None, "min_m": None, "max_m": None,
              "valid_count": 0, "sample_count": EDGE_DEPTH_SAMPLE_COUNT}
    if target_edge is None or contour is None or depth is None:
        return result
    depth = np.asarray(depth)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2 or depth.shape != mask.shape:
        result["status"] = "RGB/depth size mismatch"
        return result
    selected_mask = np.zeros_like(mask)
    cv2.drawContours(selected_mask, [contour], -1, 255, cv2.FILLED)
    selected_mask = cv2.bitwise_and(selected_mask, mask)
    kernel_size = 2 * DEPTH_INSET_PX + 1
    interior = cv2.erode(selected_mask, np.ones((kernel_size, kernel_size), np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=0) > 0
    valid = interior & np.isfinite(depth) & (depth > 0)
    height, width = depth.shape
    p1, p2 = np.array(target_edge["p1"]), np.array(target_edge["p2"])
    for name, point in (("p1", p1), ("p2", p2)):
        u, v = map(int, point)
        if 0 <= u < width and 0 <= v < height:
            value = float(depth[v, u])
            if np.isfinite(value) and value > 0:
                result[f"{name}_raw_m"] = value
    for t in np.linspace(0.0, 1.0, EDGE_DEPTH_SAMPLE_COUNT):
        u, v = np.rint(p1 + t * (p2 - p1)).astype(int)
        radius = DEPTH_SEARCH_RADIUS_PX
        x0, x1 = max(0, u - radius), min(width, u + radius + 1)
        y0, y1 = max(0, v - radius), min(height, v + radius + 1)
        ys, xs = np.nonzero(valid[y0:y1, x0:x1])
        sample = {"edge_uv": (int(u), int(v)), "surface_uv": None, "depth_m": None}
        if len(xs):
            xs, ys = xs + x0, ys + y0
            nearest = np.argmin((xs - u) ** 2 + (ys - v) ** 2)
            su, sv = int(xs[nearest]), int(ys[nearest])
            r = DEPTH_PATCH_RADIUS_PX
            px0, px1 = max(0, su - r), min(width, su + r + 1)
            py0, py1 = max(0, sv - r), min(height, sv + r + 1)
            values = depth[py0:py1, px0:px1][valid[py0:py1, px0:px1]]
            sample["surface_uv"] = (su, sv)
            sample["depth_m"] = float(np.median(values))
        result["samples"].append(sample)
    # Keep original nearby medians for diagnostics. For 3D use the depth of
    # the *original edge pixel ray* intersecting a robust measured local plane.
    for sample in result['samples']:
        sample['nearby_median_m'] = sample['depth_m']
    result['p1_nearby_median_m'] = result['samples'][0]['depth_m']
    result['p2_nearby_median_m'] = result['samples'][-1]['depth_m']
    try:
        ray_depth, diagnostics = fit_edge_surface_depth(depth, valid, p1, p2)
        depths = ray_depth([sample['edge_uv'] for sample in result['samples']])
        result.update(diagnostics)
        for sample,value in zip(result['samples'],depths):
            sample['depth_m'] = float(value)
        result.update(status='surface fit ok', valid_count=len(depths),
                      p1_surface_m=float(depths[0]), p2_surface_m=float(depths[-1]),
                      median_m=float(np.median(depths)), min_m=float(np.min(depths)),max_m=float(np.max(depths)))
    except ValueError as error:
        result['status'] = f'depth surface rejected: {error}'
        for sample in result['samples']:
            sample['depth_m'] = None
    return result


def format_depth(value):
    return "N/A" if value is None else f"{value:.4f} m"


def draw_detection_overlay(rgb_frame, contour, center, edges, target_edge, edge_depth=None):
    """Make a separate BGR display copy; never draw on the raw RGB frame."""
    display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
    if contour is not None:
        cv2.drawContours(display_frame, [contour], -1, (255, 0, 255), 2)
    for edge in edges:
        cv2.line(display_frame, edge["p1"], edge["p2"], (255, 255, 0), 2, cv2.LINE_AA)
    text = ["Green cube not detected - check Green Mask / HSV"]
    if center is not None:
        cv2.drawMarker(display_frame, center, (255, 255, 255), cv2.MARKER_CROSS, 18, 2)
        text = [f"Cube center: {center}", "No supported RGB line - check occlusion / Canny"]
    if target_edge is not None:
        p1, p2 = target_edge["p1"], target_edge["p2"]
        cv2.line(display_frame, p1, p2, (0, 0, 255), 5, cv2.LINE_AA)
        for label, point in (("p1", p1), ("p2", p2)):
            cv2.circle(display_frame, point, 6, (0, 255, 255), -1, cv2.LINE_AA)
            cv2.putText(display_frame, label, (point[0] + 8, max(15, point[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2, cv2.LINE_AA)
        text = [f"Cube center: {center}", f"p1 = {p1}   p2 = {p2}",
                f"Length: {target_edge['length_px']:.2f} px",
                f"Candidate: {target_edge['kind']}"]
    if target_edge is not None and edge_depth is not None:
        for sample in edge_depth["samples"]:
            if sample["surface_uv"] is not None:
                cv2.circle(display_frame, sample["surface_uv"], 1, (255, 255, 255), -1)
        text.extend([
            "Depth: camera Z (m), measured surface / edge ray",
            f"p1 ray Z: {format_depth(edge_depth['p1_surface_m'])}",
            f"p2 ray Z: {format_depth(edge_depth['p2_surface_m'])}",
            f"Edge median Z: {format_depth(edge_depth['median_m'])}",
            f"Range: {format_depth(edge_depth['min_m'])} - {format_depth(edge_depth['max_m'])}",
            f"Valid: {edge_depth['valid_count']}/{edge_depth['sample_count']} | {edge_depth['status']}",
        ])
        if 'surface_rms_m' in edge_depth:
            text.append(f"Surface support: {edge_depth['surface_inlier_ratio']:.0%}; "
                        f"RMS: {edge_depth['surface_rms_m'] * 1000:.2f} mm")
    for index, line in enumerate(text):
        origin = (10, 24 + index * 24)
        cv2.putText(display_frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(display_frame, line, origin, cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return display_frame


class CubeEdgeDetector:
    """RGB/depth detector implementing the existing simulation loop callbacks."""

    def __init__(self, camera=None, **_scene_arguments):
        global CURRENT_DETECTOR
        CURRENT_DETECTOR = self
        self.camera = camera
        self.stage = _scene_arguments.get("stage")
        self.robot_base_path = _scene_arguments.get("robot_base_path")
        self.tool_path = _scene_arguments.get("tool_path")
        self.robot = None
        self.world = None
        self._path_status = None
        self._preview_requested = False
        self._execute_requested = False
        self._stop_requested = False
        self.path_plan = None
        self.executor = None
        self.captured_intrinsics = None
        self.captured_world_base = None
        self.captured_camera_gt = None
        self.captured_cube_world = None
        self.path_evaluation = None
        self._evaluation_status = None
        self.calibration_source = "OpenCV"
        self._source_status = None
        self.opencv_calibration = None
        self.lrbo_calibration = None
        self._detect_requested = False
        self._frozen = False
        self._control_window = None
        self._detection_status = None
        self._comparison_status = None
        self._comparison = None
        self._opencv_path_model = None
        self._lrbo_path_model = None
        self.edge_depth = None
        self.rgb_frame = None
        self.display_frame = None
        self.green_mask = None
        self.target_edge = None
        self._last_detection_time = -float("inf")
        self._last_rendering_frame = None
        self._image_windows = {}
        self._image_providers = {}

    def request_detection(self):
        if self.executor is not None and self.executor.active:
            self._set_path_status("Stop execution before detecting another edge")
            return
        self._invalidate_path()
        self._detect_requested = True
        if self._detection_status is not None:
            self._detection_status.text = "Waiting for camera frame; one detection requested"

    def clear_detection(self):
        if self.executor is not None and self.executor.active:
            self._set_path_status("Stop execution before clearing the snapshot")
            return
        self._invalidate_path()
        self._detect_requested = False
        self._frozen = False
        self.target_edge = None
        self.edge_depth = None
        self.green_mask = None
        if self._detection_status is not None:
            self._detection_status.text = "Live preview; press DETECT ONCE to capture"

    def update_detection(self, rgba, show_window=True):
        # Read both annotators from one camera acquisition snapshot. The host
        # already enables distance_to_image_plane during camera initialization.
        if self._frozen and not self._detect_requested:
            return
        now = time.monotonic()
        if not self._detect_requested and now - self._last_detection_time < 1.0 / DETECTION_HZ:
            return
        depth = None
        rendering_frame = None
        if self.camera is not None:
            frame = self.camera.get_current_frame()
            rendering_frame = frame.get("rendering_frame")
            if (not self._detect_requested and rendering_frame is not None
                    and rendering_frame == self._last_rendering_frame):
                return
            # Installed Isaac 6.1 stores the RGBA annotator under "rgb";
            # older camera implementations may expose "rgba" instead.
            snapshot_rgba = frame.get("rgb")
            if snapshot_rgba is None:
                snapshot_rgba = frame.get("rgba")
            if snapshot_rgba is not None:
                rgba = snapshot_rgba
                depth = frame.get("distance_to_image_plane")
        image = np.asarray(rgba)
        if image.ndim != 3 or image.shape[2] < 3 or image.size == 0:
            return
        # Isaac get_rgba() is RGBA uint8. Keep independent raw RGB and overlay.
        rgb_frame = image[..., :3].astype(np.uint8, copy=True)
        if not self._detect_requested:
            # Live raw preview only. No segmentation, Hough or depth sampling.
            self.rgb_frame = rgb_frame
            self.display_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.putText(self.display_frame, "LIVE - press DETECT ONCE to freeze a result",
                        (10, 24), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1, cv2.LINE_AA)
            self._last_detection_time = now
            self._last_rendering_frame = rendering_frame
            if show_window:
                self._show_image(RGB_WINDOW_NAME, cv2.cvtColor(self.display_frame, cv2.COLOR_BGR2RGBA))
                if self.green_mask is None:
                    self._show_image(MASK_WINDOW_NAME, cv2.cvtColor(
                        np.zeros(rgb_frame.shape[:2], np.uint8), cv2.COLOR_GRAY2RGBA))
            return
        if self.camera is not None:
            self.captured_intrinsics = np.asarray(self.camera.get_intrinsics_matrix(), dtype=float).copy()
        if self.stage is not None:
            self.captured_world_base = lrbo.SCENE.get_prim_world_matrix(self.stage, self.robot_base_path).copy()
            position, orientation = self.camera.get_world_pose(camera_axes="ros")
            world_camera = lrbo.SCENE._matrix_from_rt(
                lrbo.SCENE.quaternion_wxyz_to_matrix(orientation), position)
            self.captured_camera_gt = np.linalg.inv(self.captured_world_base) @ world_camera
            self.captured_cube_world = lrbo.SCENE.get_prim_world_matrix(self.stage, "/GreenCube").copy()
        mask, contour, center = detect_green_cube(rgb_frame)
        edges = extract_edges(rgb_frame, mask, contour, depth)
        target_edge = select_target_edge(edges)
        self.edge_depth = sample_edge_depth(depth, mask, contour, target_edge)
        display_frame = draw_detection_overlay(rgb_frame, contour, center, edges, target_edge,
                                               self.edge_depth)
        self.rgb_frame, self.display_frame = rgb_frame, display_frame
        self.green_mask, self.target_edge = mask, target_edge
        self._last_detection_time = now
        self._last_rendering_frame = rendering_frame
        self._detect_requested = False
        self._frozen = True
        if self._detection_status is not None:
            self._detection_status.text = (
                "FROZEN snapshot: edge/depth captured; DETECT ONCE to replace"
                if target_edge is not None else "FROZEN snapshot: no supported edge; reposition then DETECT ONCE")
        if target_edge is not None:
            p1, p2 = target_edge["p1"], target_edge["p2"]
            print(f"Cube center: {center}\nSelected edge: {p1} -> {p2}\n"
                  f"Length: {target_edge['length_px']:.2f} px\n"
                  f"Candidate: {target_edge['kind']}; RGB support: {target_edge['support']:.2f}\n"
                  f"p1 = {p1}\np2 = {p2}\n"
                  f"Depth: camera optical Z (m); measured local plane / original ray\n"
                  f"p1 ray Z: {format_depth(self.edge_depth['p1_surface_m'])}\n"
                  f"p2 ray Z: {format_depth(self.edge_depth['p2_surface_m'])}\n"
                  f"Exact p1 pixel Z: {format_depth(self.edge_depth['p1_raw_m'])}\n"
                  f"Exact p2 pixel Z: {format_depth(self.edge_depth['p2_raw_m'])}\n"
                  f"Edge median Z: {format_depth(self.edge_depth['median_m'])}\n"
                  f"Edge Z range: {format_depth(self.edge_depth['min_m'])} - "
                  f"{format_depth(self.edge_depth['max_m'])}\n"
                  f"Valid depth samples: {self.edge_depth['valid_count']}/"
                  f"{self.edge_depth['sample_count']} ({self.edge_depth['status']})",
                  flush=True)
        else:
            print("Cube edge not detected; check Green Mask and detection parameters.",
                  flush=True)
        if self.stage is not None and self.target_edge is not None:
            self._preview_requested = True
        if show_window:
            self._show_image(RGB_WINDOW_NAME, cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGBA))
            self._show_image(MASK_WINDOW_NAME, cv2.cvtColor(mask, cv2.COLOR_GRAY2RGBA))

    def _show_image(self, title, rgba):
        """Display live camera pixels through Isaac's native UI, without HighGUI.

        Isaac's bundled OpenCV can process pixels but lacks namedWindow/imshow.
        ByteImageProvider keeps the overlays in the same Isaac application.
        """
        ui = lrbo.SCENE.ui
        height, width = rgba.shape[:2]
        if title not in self._image_providers:
            provider = ui.ByteImageProvider()
            window = ui.Window(title, width=width + 20, height=height + 40)
            with window.frame:
                ui.ImageWithProvider(provider)
            self._image_providers[title] = provider
            self._image_windows[title] = window
        self._image_providers[title].set_bytes_data(
            bytearray(np.ascontiguousarray(rgba).tobytes()), [width, height])

    def _set_path_status(self, text):
        if self._path_status is not None:
            self._path_status.text = text
        print(f"[edge path] {text}", flush=True)

    def _invalidate_path(self):
        self.path_plan = None
        self.path_evaluation = None
        if self._evaluation_status is not None:
            self._evaluation_status.text = "DETECT ONCE / BUILD PATH to compare with ideal cube edge"
        self._preview_requested = False
        self._execute_requested = False
        if self.stage is not None:
            self.stage.RemovePrim("/CubeEdgePathPreview")

    def request_preview(self):
        self._preview_requested = True

    def request_execution(self):
        self._execute_requested = True

    def request_stop(self):
        self._stop_requested = True

    def select_calibration_source(self, source):
        if self.executor is not None and self.executor.active:
            self._set_path_status("Stop execution before switching calibration source")
            return
        if source != self.calibration_source:
            self._execute_requested = False
            self._preview_requested = False
            self._invalidate_path()
            self.calibration_source = source
        if self._source_status is not None:
            self._source_status.text = f"Path calibration: {source} (saved parameters); build preview before execution"

    def load_selected_calibration(self):
        model = self._opencv_path_model if self.calibration_source == "OpenCV" else self._lrbo_path_model
        return load_calibration(model.get_value_as_string())

    def build_path_preview(self):
        if self.executor is not None and self.executor.active:
            raise ValueError("Cannot replace plan during execution")
        self._invalidate_path()
        if self.stage is None or self.target_edge is None or self.captured_intrinsics is None:
            raise ValueError("DETECT ONCE must capture a valid edge and camera intrinsics first")
        selected = self.load_selected_calibration()
        current_base = lrbo.SCENE.get_prim_world_matrix(self.stage, self.robot_base_path)
        if not np.allclose(current_base, self.captured_world_base, atol=1e-5, rtol=0):
            raise ValueError("Robot base changed since capture; detect again")
        metadata = selected["metadata"]
        resolution = metadata.get("camera_resolution")
        if resolution and resolution != f"{self.rgb_frame.shape[1]} x {self.rgb_frame.shape[0]}":
            raise ValueError("Saved calibration RGB resolution differs from detection")
        gt = selected["ground_truth"]
        if gt is None or not np.allclose(gt, self.captured_camera_gt, atol=1e-5, rtol=0):
            raise ValueError("Saved calibration base/camera pose cannot be verified against this scene")
        model = URDFModel(lrbo.SCENE.SOURCE_URDF)
        offset, axis, roll, _ = model.closed_tip(lrbo.SCENE.ROBOTIQ_JOINT_MULTIPLIERS)
        if TCP_TO_TIP_OFFSET is not None:
            offset = np.asarray(TCP_TO_TIP_OFFSET, dtype=float)
        if TCP_TIP_AXIS is not None:
            axis = unit(TCP_TIP_AXIS, "configured tool axis")
        tabletop = lrbo.SCENE.TABLE_HEIGHT + lrbo.SCENE.TABLE_THICKNESS
        xy = np.asarray(lrbo.SCENE.WORKSPACE_CENTER[:2], dtype=float)
        half = np.array([lrbo.SCENE.TABLE_WIDTH, lrbo.SCENE.TABLE_DEPTH]) / 2
        tip_bounds = (np.r_[xy-half+.005, tabletop+.002], np.r_[xy+half-.005, tabletop+.80])
        tcp_bounds = (np.r_[xy-half-TCP_WORKSPACE_MARGIN, tabletop+.002],
                      np.r_[xy+half+TCP_WORKSPACE_MARGIN, tabletop+1.20])
        # Choose the tool projection toward the cube, so the tool body lies on
        # the outward side of the edge. Geometry is checked before execution.
        cube_world = lrbo.SCENE.get_prim_world_matrix(self.stage, "/GreenCube")[:3,3]
        cube_base = transform_points(np.linalg.inv(current_base), cube_world)
        from edge_tip_path import back_project
        midpoint_camera = np.mean([
            back_project(self.target_edge['p1'], self.edge_depth['p1_surface_m'], self.captured_intrinsics),
            back_project(self.target_edge['p2'], self.edge_depth['p2_surface_m'], self.captured_intrinsics)], axis=0)
        toward = cube_base - transform_points(selected['matrix'], midpoint_camera)
        plan = build_edge_plan(self.target_edge, self.edge_depth, self.captured_intrinsics,
                              selected['matrix'], current_base, offset, axis, roll, TABLE_NORMAL,
                              EDGE_CLEARANCE, TOOL_TILT_DEG, WAYPOINT_SPACING, toward,
                              tip_bounds, tcp_bounds, DEPTH_LINE_CONSISTENCY_M)
        plan['calibration_file'] = selected['path']
        plan['calibration_source'] = self.calibration_source
        plan['camera_gt'] = self.captured_camera_gt.copy()
        plan['cube_world'] = lrbo.SCENE.get_prim_world_matrix(self.stage, "/GreenCube").copy()
        if not np.allclose(plan['cube_world'], self.captured_cube_world, atol=1e-5, rtol=0):
            raise ValueError("Cube moved since snapshot; detect again")
        draw_plan(lrbo.SCENE, self.stage, plan)
        # USD Cube.GetSize() is the unscaled local size (FixedCuboid uses 1.0).
        # cube_world already contains the physical .12 m scale.
        local_size = float(lrbo.SCENE.UsdGeom.Cube(self.stage.GetPrimAtPath('/GreenCube')).GetSizeAttr().Get())
        try:
            self.path_evaluation = evaluate_path(plan, plan['cube_world'], local_size,
                plan['camera_gt'], self.captured_intrinsics, axis, roll, TABLE_NORMAL,
                EDGE_CLEARANCE, TOOL_TILT_DEG)
        except ValueError as error:
            self.path_evaluation = {'matched': False, 'reason': str(error)}
        draw_ideal_reference(self.stage, plan, self.path_evaluation)
        evaluation_summary = evaluation_text(self.path_evaluation)
        if self._evaluation_status is not None:
            self._evaluation_status.text = evaluation_summary
        print('[ideal path comparison]\n' + evaluation_summary, flush=True)
        if self.path_evaluation['matched']:
            for index, label in enumerate(('P1','P2')):
                print(f"{label} camera estimate - ideal xyz (mm): "
                      f"{self.path_evaluation['endpoint_delta_est_minus_ideal_mm'][index]}", flush=True)
        evaluation_dir = COMPARISON_DIR / 'edge_paths'
        evaluation_dir.mkdir(parents=True, exist_ok=True)
        evaluation_file = evaluation_dir / f'edge_path_{lrbo._timestamp()}.json'
        self.path_evaluation['calibration_source'] = self.calibration_source
        self.path_evaluation['calibration_file'] = selected['path']
        evaluation_file.write_text(json.dumps(self.path_evaluation, indent=2) + '\n')
        print(f'Path comparison saved: {evaluation_file}', flush=True)
        self.path_plan = plan
        if self.executor is not None:
            self.executor.close()
            self.executor = None
        if self.robot is not None:
            self.executor = SlowEdgeExecutor(lrbo.SCENE, self.stage, self.robot, self.world,
                self.robot_base_path, self.tool_path, model, sys.modules[__name__], self._set_path_status)
        for name, value in (("P1/P2 pixel", plan['pixels']), ("P1/P2 camera (m)", plan['camera']),
                            ("P1/P2 base (m)", plan['base']), ("edge direction", plan['edge_direction']),
                            ("tool direction", plan['tool_direction']), ("TCP local axis", axis),
                            ("TCP offset d_tcp_tip (m)", plan['offset']),
                            ("R_base_tcp", plan['rotation']), ("Quaternion wxyz", plan['quaternion_wxyz'])):
            print(f"{name}: {value}", flush=True)
        self._set_path_status(f"Preview ({self.calibration_source}): red=edge, cyan=tip, orange=TCP; {len(plan['tip'])} waypoints. "
                              f"Final fixed-tip turn to vertical: blue TCP arc. "
                              f"EXECUTE_ROBOT={EXECUTE_ROBOT}; closed-gripper tip offset={offset.round(5)} m")

    def process_commands(self):
        if self._stop_requested:
            self._stop_requested = False
            self._execute_requested = False
            if self.executor is not None:
                self.executor.stop("Stopped by user; holding current joints")
        if self._preview_requested:
            self._preview_requested = False
            try:
                self.build_path_preview()
            except Exception as error:
                self._set_path_status(f"Preview rejected: {error}")
        if self._execute_requested:
            self._execute_requested = False
            try:
                if not EXECUTE_ROBOT:
                    self._set_path_status("PREVIEW ONLY: EXECUTE_ROBOT=False; robot and gripper unchanged")
                elif self.executor is None or self.path_plan is None:
                    raise ValueError("Build valid path preview first")
                else:
                    selected = self.load_selected_calibration()
                    if self.calibration_source != self.path_plan['calibration_source'] or selected['path'] != self.path_plan['calibration_file'] or not np.allclose(
                            selected['matrix'], self.path_plan['base_camera'], atol=1e-10, rtol=0):
                        raise ValueError("Selected calibration parameters changed; rebuild preview")
                    position, orientation = self.camera.get_world_pose(camera_axes="ros")
                    world_camera = lrbo.SCENE._matrix_from_rt(
                        lrbo.SCENE.quaternion_wxyz_to_matrix(orientation), position)
                    current_base = lrbo.SCENE.get_prim_world_matrix(self.stage, self.robot_base_path)
                    current_camera_gt = np.linalg.inv(current_base) @ world_camera
                    if not np.allclose(current_camera_gt, self.path_plan['camera_gt'], atol=1e-5, rtol=0):
                        raise ValueError("Camera moved since preview; detect again")
                    cube_world = lrbo.SCENE.get_prim_world_matrix(self.stage, '/GreenCube')
                    if not np.allclose(cube_world, self.path_plan['cube_world'], atol=1e-5, rtol=0):
                        raise ValueError("Cube moved since detection; detect again")
                    self.executor.start(self.path_plan)
            except Exception as error:
                if self.executor is not None and self.executor.active:
                    self.executor.stop(f"Execution stopped: {error}")
                else:
                    self._set_path_status(f"Execution rejected: {error}")
        if self.executor is not None:
            self.executor.tick()
        # Host loop compatibility only: discard legacy calibration requests.
        command_queue = lrbo.SCENE.CALIBRATION_COMMAND_QUEUE
        while True:
            try:
                command_queue.get_nowait()
            except lrbo.SCENE.queue.Empty:
                break

    def compare_parameters(self):
        if self.executor is not None and self.executor.active:
            self._set_path_status("Stop execution before replacing calibration")
            return
        self._invalidate_path()
        try:
            opencv = load_calibration(self._opencv_path_model.get_value_as_string())
            other = load_calibration(self._lrbo_path_model.get_value_as_string())
            self._comparison = compare_calibrations(opencv, other)
            self.opencv_calibration = opencv
            self.lrbo_calibration = other
            text = comparison_text(self._comparison)
            self._comparison_status.text = text
            print(text, flush=True)
            for name in ("opencv", "lrbo"):
                print(f"T_base_camera ({name}):\n" + np.array2string(
                    np.asarray(self._comparison[f"T_base_camera_{name}"]), precision=8), flush=True)
        except Exception as error:
            self._comparison = None
            self.opencv_calibration = None
            self.lrbo_calibration = None
            self._comparison_status.text = f"Comparison unavailable: {error}"
            print(self._comparison_status.text, flush=True)

    def save_comparison(self):
        if self._comparison is None:
            self._comparison_status.text = "Compare successfully before saving a report"
            return
        COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
        path = COMPARISON_DIR / f"opencv_lrbo_{lrbo._timestamp()}.json"
        path.write_text(json.dumps(self._comparison, indent=2) + "\n")
        self._comparison_status.text = comparison_text(self._comparison) + f"\nSaved: {path.name}"
        print(f"Comparison saved: {path}", flush=True)

    def create_control_window(self):
        ui = lrbo.SCENE.ui
        window = ui.Window("Cube detection / OpenCV vs LRBO", width=820, height=730)
        self._control_window = window
        with window.frame:
            with ui.VStack(spacing=6, height=0):
                with ui.HStack(height=30, spacing=8):
                    ui.Button("USE OpenCV", clicked_fn=lambda: self.select_calibration_source("OpenCV"))
                    ui.Button("USE LRBO", clicked_fn=lambda: self.select_calibration_source("LRBO"))
                self._source_status = ui.Label("Path calibration: OpenCV (saved parameters)", height=26)
                with ui.HStack(height=34, spacing=8):
                    ui.Button("DETECT ONCE", clicked_fn=self.request_detection)
                    ui.Button("CLEAR / LIVE PREVIEW", clicked_fn=self.clear_detection)
                self._detection_status = ui.Label("Live preview; press DETECT ONCE to capture", height=24)
                ui.Label("Frozen overlay / depth belongs to the captured image. D435 RGB viewport stays live.", height=24)
                with ui.HStack(height=34, spacing=8):
                    ui.Button("BUILD / PREVIEW 3D PATH", clicked_fn=self.request_preview)
                    ui.Button("EXECUTE PATH", clicked_fn=self.request_execution)
                    ui.Button("STOP", clicked_fn=self.request_stop)
                self._path_status = ui.Label(f"EXECUTE_ROBOT={EXECUTE_ROBOT}; press EXECUTE PATH to run once. No automatic movement.", height=48, word_wrap=True)
                ui.Separator(height=4)
                self._evaluation_status = ui.Label("DETECT ONCE / BUILD PATH to compare with ideal cube edge",
                                                   height=150, word_wrap=True)
                ui.Label("Ideal references: magenta=edge, blue=tip, green=TCP; gray=full edge", height=24)
                ui.Label("OpenCV saved calibration (YAML / NPY)", height=22)
                self._opencv_path_model = ui.StringField(height=24).model
                self._opencv_path_model.set_value(default_calibration_file(
                    "OPENCV_CALIBRATION_FILE", lrbo.WORKSPACE_ROOT / "isaac_sim" / "calibration", "*.yaml"))
                ui.Label("LRBO saved calibration (JSON / NPY)", height=22)
                self._lrbo_path_model = ui.StringField(height=24).model
                self._lrbo_path_model.set_value(default_calibration_file(
                    "LRBO_CALIBRATION_FILE", lrbo.RESULT_DIR, "*.json"))
                with ui.HStack(height=34, spacing=8):
                    ui.Button("LOAD / COMPARE SAVED", clicked_fn=self.compare_parameters)
                    ui.Button("SAVE COMPARISON", clicked_fn=self.save_comparison)
                self._comparison_status = ui.Label("Select exports then compare", height=190, word_wrap=True)
        self.compare_parameters()
        return window

    def close(self):
        if self.executor is not None:
            self.executor.close()
            self.executor = None
        if self._control_window is not None:
            self._control_window.destroy()
            self._control_window = None
        for window in self._image_windows.values():
            window.destroy()
        self._image_windows.clear()
        self._image_providers.clear()


def _start_detection_control_gui(scene):
    """Keep manual Home/joint/gripper controls; disable calibration and recording."""
    original_label, original_button = scene.ui.Label, scene.ui.Button

    def label_without_calibration(*args, **kwargs):
        widget = original_label(*args, **kwargs)
        value = str(args[0] if args else kwargs.get("text", ""))
        if ("ChArUco" in value or value.startswith(
                ("Samples:", "Isaac Sim GT error:", "Solver:", "PARK:", "ANDREFF:", "D435 Recording", "Recording:"))):
            widget.visible = False
        return widget

    def button_without_calibration(text, *args, **kwargs):
        widget = original_button(text, *args, **kwargs)
        if str(text) in {"CAPTURE", "SOLVE", "CLEAR", "SAVE", "START RECORDING", "STOP RECORDING"}:
            widget.visible = False
        return widget

    scene.ui.Label, scene.ui.Button = label_without_calibration, button_without_calibration
    try:
        return scene._cube_original_start_joint_control_gui()
    finally:
        scene.ui.Label, scene.ui.Button = original_label, original_button


def main():
    # Reuse scene configuration and manual controls; do not create a calibrator.
    scene = lrbo._load_scene_module()
    lrbo.SCENE = scene
    create_workspace = scene.create_workspace
    create_articulation = scene.create_ur5e_articulation
    context = {}

    def create_articulation_with_context(root, world):
        robot = create_articulation(root, world)
        context.update(robot=robot, world=world)
        return robot

    scene.create_ur5e_articulation = create_articulation_with_context
    original_joint_commands = scene.process_joint_commands
    original_gripper_commands = scene.process_gripper_commands

    def guarded_joint_commands(robot, indices):
        if CURRENT_DETECTOR is not None and CURRENT_DETECTOR.executor is not None and CURRENT_DETECTOR.executor.active:
            while not scene.JOINT_COMMAND_QUEUE.empty():
                scene.JOINT_COMMAND_QUEUE.get_nowait()
        else:
            original_joint_commands(robot, indices)

    def guarded_gripper_commands(robot, indices, multipliers):
        if CURRENT_DETECTOR is not None and CURRENT_DETECTOR.executor is not None and CURRENT_DETECTOR.executor.active:
            while not scene.GRIPPER_COMMAND_QUEUE.empty():
                scene.GRIPPER_COMMAND_QUEUE.get_nowait()
        else:
            original_gripper_commands(robot, indices, multipliers)

    scene.process_joint_commands = guarded_joint_commands
    scene.process_gripper_commands = guarded_gripper_commands

    def create_workspace_with_cube(world, stage):
        result = create_workspace(world, stage)
        spawn_random_green_cube(world, stage)
        return result

    scene.create_workspace = create_workspace_with_cube
    scene.create_charuco_board_definition = lambda: (None, None)
    scene.generate_charuco_texture = lambda _board: None
    scene.create_charuco_board_prim = lambda _stage, _tool_path: "RGB edge detection; no calibration target"
    scene.initialize_depth_window = lambda: None
    # Host loop expects recorder.active/stop; this adapter never creates a writer.
    class DisabledRecorder:
        active = False

        def __init__(self, *args, **kwargs):
            pass

        def stop(self):
            pass

    scene.D435Recorder = DisabledRecorder
    scene.process_record_commands = lambda recorder: None
    scene.push_record_status = lambda status: None
    scene.show_depth = lambda depth: None
    scene.EyeToHandCalibrator = CubeEdgeDetector
    scene._cube_original_start_joint_control_gui = scene.start_joint_control_gui

    def start_combined_gui():
        joint_gui = _start_detection_control_gui(scene)
        CURRENT_DETECTOR.robot = context['robot']
        CURRENT_DETECTOR.world = context['world']
        CURRENT_DETECTOR.create_control_window()
        return joint_gui

    scene.start_joint_control_gui = start_combined_gui
    # Home/manual controls remain active while the optional executor is idle.
    scene.main()


if __name__ == "__main__":
    main()
