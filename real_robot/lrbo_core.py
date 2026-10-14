"""LRBO point-cloud registration and frame conversion helpers."""

from __future__ import annotations

import re

import numpy as np

LRBO_MODEL_SCALE = 5.0
# Isaac depth is already metric; the notebook correction was camera-specific.
LRBO_SOURCE_SCALE = LRBO_MODEL_SCALE
LRBO_REGISTRATION_VOXEL_M = 0.02
LRBO_ICP_THRESHOLD = 0.1
LRBO_ICP_MAX_ITERATIONS = 1000


def _image_2d(image, expected_shape, name):
    array = np.asarray(image)
    if array.ndim == 3:
        if array.shape[-1] != 1:
            raise ValueError(f"{name} must be a single-channel image.")
        array = array[..., 0]
    if array.shape != expected_shape:
        if array.T.shape == expected_shape:
            array = array.T
        else:
            raise ValueError(
                f"{name} shape {array.shape} does not match depth shape {expected_shape}."
            )
    return array


def _flatten_label_values(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield str(key)
            yield from _flatten_label_values(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _flatten_label_values(item)
    else:
        yield str(value)


def instance_ids_for_prim_tokens(segmentation_info, prim_tokens):
    """Find instance IDs whose annotator labels contain any requested prim name."""
    id_to_labels = segmentation_info.get("idToLabels", {})
    requested = {token.lower() for token in prim_tokens}
    selected = []
    for instance_id, labels in id_to_labels.items():
        for value in _flatten_label_values(labels):
            parts = re.split(r"[/\\\s\"':]+", value.lower())
            # A nested arm link must not inherit its ancestor's base ROI membership.
            links = [part for part in parts if part.endswith("_link")
                     or part == "base_link_inertia"]
            if links and links[-1] in requested:
                selected.append(int(instance_id))
                break
    return np.asarray(sorted(set(selected)), dtype=np.uint32)


def depth_mask_to_camera_points(depth, instance_ids, camera_matrix, selected_ids):
    """Back-project masked distance-to-image-plane pixels into ROS optical coordinates."""
    depth_array = np.asarray(depth, dtype=np.float64)
    if depth_array.ndim == 3:
        if depth_array.shape[-1] != 1:
            raise ValueError("Depth must be a single-channel image.")
        depth_array = depth_array[..., 0]
    if depth_array.ndim != 2:
        raise ValueError(f"Expected a 2D depth image, got {depth_array.shape}.")

    ids_array = _image_2d(instance_ids, depth_array.shape, "instance segmentation")
    matrix = np.asarray(camera_matrix, dtype=np.float64)
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError("Camera intrinsics must be a finite 3x3 matrix.")
    if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
        raise ValueError("Camera focal lengths must be positive.")

    mask = np.isin(ids_array, np.asarray(selected_ids).reshape(-1))
    mask &= np.isfinite(depth_array) & (depth_array > 0.0)
    rows, columns = np.nonzero(mask)
    z = depth_array[rows, columns]
    x = (columns - matrix[0, 2]) * z / matrix[0, 0]
    y = (rows - matrix[1, 2]) * z / matrix[1, 1]
    return np.column_stack((x, y, z)).astype(np.float32), int(mask.sum())


def lrbo_transform_to_base_camera(T_model_from_source):
    """Convert camera-to-model registration to model-from-camera in meters.

    Both clouds use the same 5x scale. Thus R is unchanged and t is divided
    by five. The notebook's inverse instead describes camera-from-model.
    """
    transform = validate_rigid_transform(T_model_from_source)
    result = transform.copy()
    result[:3, 3] /= LRBO_MODEL_SCALE
    return result


def validate_rigid_transform(transform):
    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Registration must return a finite 4x4 transform.")
    rotation = transform[:3, :3]
    if (not np.allclose(transform[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-4)
            or not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-4)):
        raise ValueError("Registration must be a rigid transform with a proper rotation.")
    return transform


def _voxel_downsample(points, voxel_size, open3d_module):
    point_cloud = open3d_module.geometry.PointCloud()
    point_cloud.points = open3d_module.utility.Vector3dVector(
        np.asarray(points, dtype=np.float64)
    )
    downsampled = point_cloud.voxel_down_sample(float(voxel_size))
    return np.asarray(downsampled.points, dtype=np.float32)


def run_lrbo_registration(source_camera_points, target_base_points, predator_register,
                          initial_candidates=()):
    """Run LRBO PREDATOR initialization, ICP refinement, and notebook frame conversion."""
    try:
        import open3d as o3d
    except ImportError as error:
        raise RuntimeError(
            "Open3D is required by the LRBO ICP refinement but is not installed."
        ) from error

    source = np.asarray(source_camera_points, dtype=np.float32)
    target = np.asarray(target_base_points, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] != 3 or len(source) < 100:
        raise ValueError("The captured base ROI must contain at least 100 3D points.")
    if target.ndim != 2 or target.shape[1] != 3 or len(target) < 100:
        raise ValueError("The LRBO base model must contain at least 100 3D points.")

    if not np.all(np.isfinite(source)) or not np.all(np.isfinite(target)):
        raise ValueError("Point clouds must contain only finite coordinates.")

    source_registered = source * LRBO_SOURCE_SCALE
    source_registered = _voxel_downsample(
        source_registered, LRBO_REGISTRATION_VOXEL_M, o3d
    )
    target_registered = _voxel_downsample(
        target, LRBO_REGISTRATION_VOXEL_M, o3d
    )
    if len(source_registered) < 100 or len(target_registered) < 100:
        raise ValueError("Voxel filtering left too few points for LRBO registration.")

    initial_result = predator_register(source_registered, target_registered)
    if isinstance(initial_result, tuple):
        initial_result = initial_result[0]
    initial_transform = validate_rigid_transform(initial_result)

    source_cloud = o3d.geometry.PointCloud()
    target_cloud = o3d.geometry.PointCloud()
    source_cloud.points = o3d.utility.Vector3dVector(source_registered.astype(np.float64))
    target_cloud.points = o3d.utility.Vector3dVector(target_registered.astype(np.float64))
    registration_api = getattr(o3d, "pipelines", o3d).registration
    refinements = []
    for candidate in (initial_transform, *initial_candidates):
        candidate = validate_rigid_transform(candidate)
        refinement = registration_api.registration_icp(
            source_cloud, target_cloud, LRBO_ICP_THRESHOLD, candidate,
            registration_api.TransformationEstimationPointToPoint(),
            registration_api.ICPConvergenceCriteria(max_iteration=LRBO_ICP_MAX_ITERATIONS),
        )
        refinements.append(refinement)
    # Prefer more covered source points, then the smaller geometric residual.
    refinement = max(refinements, key=lambda item: (float(item.fitness),
                                                    -float(item.inlier_rmse)))
    refined_transform = np.asarray(refinement.transformation, dtype=np.float64)
    return {
        "T_model_from_source_scaled": refined_transform,
        "T_base_camera": lrbo_transform_to_base_camera(refined_transform),
        "fitness": float(refinement.fitness),
        "inlier_rmse": float(refinement.inlier_rmse),
        "inlier_rmse_m": float(refinement.inlier_rmse) / LRBO_MODEL_SCALE,
        "source_points": int(len(source_registered)),
        "target_points": int(len(target_registered)),
    }


def sample_triangle_surface(triangles, count=40000, seed=0):
    """Area-weighted CAD sampling, independent of camera pose and captured depth."""
    triangles = np.asarray(triangles, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("Expected Nx3x3 mesh triangles")
    areas = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0],
                                    triangles[:, 2] - triangles[:, 0]), axis=1)
    valid = np.isfinite(areas) & (areas > 1e-12)
    triangles, areas = triangles[valid], areas[valid]
    if not len(triangles):
        raise ValueError("Robot ROI mesh contains no nondegenerate triangles")
    rng = np.random.default_rng(seed)
    chosen = triangles[rng.choice(len(triangles), count, p=areas / areas.sum())]
    u = np.sqrt(rng.random(count))[:, None]
    v = rng.random(count)[:, None]
    return ((1-u)*chosen[:, 0] + u*(1-v)*chosen[:, 1] + u*v*chosen[:, 2]).astype(np.float32)
