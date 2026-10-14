"""Eye-to-hand transforms use metres: T_a_b maps b coordinates into a."""
import cv2
import numpy as np


def validate_transform(value):
    value = np.asarray(value, dtype=float)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError('Expected finite 4x4 transform')
    r = value[:3, :3]
    if not (np.allclose(value[3], [0, 0, 0, 1]) and
            np.allclose(r.T @ r, np.eye(3), atol=1e-5) and
            np.isclose(np.linalg.det(r), 1, atol=1e-5)):
        raise ValueError('Expected rigid transform')
    return value


def solve(robot, target, method='ANDREFF'):
    robot = np.asarray([validate_transform(t) for t in robot])
    target = np.asarray([validate_transform(t) for t in target])
    if len(robot) != len(target) or len(robot) < 10:
        raise ValueError('Need at least 10 paired poses')
    # Rotations about at least two axes are necessary for observability.
    rotations = np.array([cv2.Rodrigues(t[:3, :3] @ robot[0, :3, :3].T)[0].ravel()
                          for t in robot[1:]])
    if np.linalg.svd(rotations, compute_uv=False)[1] < 0.1:
        raise ValueError('Insufficient rotation diversity; rotate about multiple axes')
    inverse = np.linalg.inv(robot)
    r, t = cv2.calibrateHandEye(
        list(inverse[:, :3, :3]), list(inverse[:, :3, 3]),
        list(target[:, :3, :3]), list(target[:, :3, 3]),
        method=getattr(cv2, 'CALIB_HAND_EYE_' + method))
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = r, t.ravel()
    validate_transform(result)
    closure = inverse @ result @ target
    xyz = closure[:, :3, 3]
    u, _, vt = np.linalg.svd(closure[:, :3, :3].mean(axis=0))
    correction = np.diag([1, 1, np.linalg.det(u @ vt)])
    mean_r = u @ correction @ vt
    angles = [np.linalg.norm(cv2.Rodrigues(x[:3, :3] @ mean_r.T)[0]) for x in closure]
    return result, dict(
        closure_translation_rms_mm=float(np.sqrt(np.mean(np.sum((xyz-xyz.mean(axis=0))**2, axis=1)))*1000),
        closure_rotation_rms_deg=float(np.rad2deg(np.sqrt(np.mean(np.square(angles))))))
