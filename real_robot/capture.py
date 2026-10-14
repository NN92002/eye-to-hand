"""Read-only ROS2 RGB-D/TF acquisition. No robot command publishers."""
import argparse
import json
from datetime import datetime
from pathlib import Path
import time
import cv2
import numpy as np
from core import solve, validate_transform


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=['charuco', 'lrbo'], required=True)
    p.add_argument('--image', default='/camera/camera/color/image_raw')
    p.add_argument('--info', default='/camera/camera/color/camera_info')
    p.add_argument('--depth', default='/camera/camera/aligned_depth_to_color/image_raw')
    p.add_argument('--base', default='base')
    p.add_argument('--tool', default='tool0')
    p.add_argument('--model', type=Path, help='LRBO Nx3 .npy CAD points in --base frame, metres')
    p.add_argument('--output', type=Path, default=Path('real_robot/data'))
    p.add_argument('--square', type=float, default=.04)
    p.add_argument('--marker', type=float, default=.03)
    p.add_argument('--squares-x', type=int, default=7)
    p.add_argument('--squares-y', type=int, default=5)
    a = p.parse_args()
    if not 0 < a.marker < a.square:
        p.error('Require 0 < marker < square (metres)')
    model = None
    if a.method == 'lrbo':
        if a.model is None:
            p.error('--model is required for LRBO')
        model = np.load(a.model, allow_pickle=False)
        if model.ndim != 2 or model.shape[1] != 3 or len(model) < 100 or not np.isfinite(model).all():
            p.error('Model must contain >=100 finite Nx3 points in metres')
    import rclpy
    from rclpy.duration import Duration
    from rclpy.time import Time
    from rclpy.qos import qos_profile_sensor_data
    from sensor_msgs.msg import Image, CameraInfo
    from cv_bridge import CvBridge
    import message_filters
    from tf2_ros import Buffer, TransformListener
    rclpy.init()
    node = rclpy.create_node('real_eye_to_hand_capture')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    bridge = CvBridge()
    state = {}
    node.create_subscription(CameraInfo, a.info, lambda msg: state.update(info=msg), qos_profile_sensor_data)
    rgb = message_filters.Subscriber(node, Image, a.image, qos_profile=qos_profile_sensor_data)
    depth = message_filters.Subscriber(node, Image, a.depth, qos_profile=qos_profile_sensor_data)
    sync = message_filters.ApproximateTimeSynchronizer([rgb, depth], 5, .03)
    sync.registerCallback(lambda color, dep: state.update(pair=(color, dep), received=time.monotonic()))
    dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
    if hasattr(cv2.aruco, 'CharucoBoard_create'):
        board = cv2.aruco.CharucoBoard_create(a.squares_x, a.squares_y, a.square, a.marker, dictionary)
        obj = board.chessboardCorners
    else:
        board = cv2.aruco.CharucoBoard((a.squares_x, a.squares_y), a.square, a.marker, dictionary)
        obj = board.getChessboardCorners()
    session = a.output / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    session.mkdir(parents=True)
    robot, targets, stamps, errors = [], [], [], []
    last_stamp = None
    print('c: capture, s: solve ChArUco, q: quit. Move with pendant, STOP before capture.')
    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=.01)
            if 'pair' not in state or 'info' not in state:
                continue
            color, dep = state['pair']
            info = state['info']
            image = bridge.imgmsg_to_cv2(color, 'bgr8').copy()
            key = cv2.waitKey(1) & 255
            cv2.imshow('D435 calibration', image)
            if key == ord('q'):
                break
            if key == ord('s') and a.method == 'charuco':
                try:
                    result, diagnostics = solve(robot, targets)
                    np.save(session / 'T_base_camera.npy', result)
                    (session / 'result.json').write_text(json.dumps(dict(
                        method='ANDREFF', base_frame=a.base, tool_frame=a.tool,
                        camera_frame=color.header.frame_id, samples=len(robot),
                        T_base_camera=result.tolist(), **diagnostics), indent=2))
                    print(result, diagnostics)
                except (ValueError, cv2.error) as e:
                    print('Solve rejected:', e)
                continue
            if key != ord('c'):
                continue
            try:
                stamp = (color.header.stamp.sec, color.header.stamp.nanosec)
                if stamp == last_stamp or time.monotonic()-state['received'] > .5:
                    raise ValueError('Duplicate or stale image')
                if info.header.frame_id != color.header.frame_id or (info.width, info.height) != (image.shape[1], image.shape[0]):
                    raise ValueError('CameraInfo must match color frame and resolution')
                if info.distortion_model not in ('plumb_bob', 'rational_polynomial'):
                    raise ValueError('Unsupported distortion model: ' + info.distortion_model)
                k = np.asarray(info.k).reshape(3, 3)
                d = np.asarray(info.d)
                if not np.isfinite(k).all() or k[0, 0] <= 0 or k[1, 1] <= 0:
                    raise ValueError('Invalid intrinsics')
                tf = buffer.lookup_transform(a.base, a.tool, Time.from_msg(color.header.stamp), timeout=Duration(seconds=.05))
                q = tf.transform.rotation
                v = np.array([q.x, q.y, q.z], dtype=float)
                norm = np.linalg.norm([q.x, q.y, q.z, q.w])
                if not np.isclose(norm, 1, atol=1e-5):
                    raise ValueError('Invalid TF quaternion')
                skew = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
                t = np.eye(4)
                t[:3, :3] = np.eye(3) + 2*q.w*skew + 2*skew@skew
                tr = tf.transform.translation
                t[:3, 3] = [tr.x, tr.y, tr.z]
                validate_transform(t)
                if a.method == 'charuco':
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    corners, ids, _ = cv2.aruco.detectMarkers(gray, dictionary)
                    if ids is None:
                        raise ValueError('No markers')
                    _, cc, ci = cv2.aruco.interpolateCornersCharuco(corners, ids, gray, board, cameraMatrix=k, distCoeffs=d)
                    if ci is None or len(ci) < 6:
                        raise ValueError('Need >=6 ChArUco corners')
                    points = obj[ci.ravel()]
                    ok, rv, tv = cv2.solvePnP(points, cc, k, d)
                    if not ok or tv[2, 0] <= 0:
                        raise ValueError('PnP failed')
                    projected = cv2.projectPoints(points, rv, tv, k, d)[0]
                    error = float(np.linalg.norm(projected.reshape(-1, 2)-cc.reshape(-1, 2), axis=1).mean())
                    if error > 1.5:
                        raise ValueError(f'Reprojection error {error:.2f}px >1.5px')
                    target = np.eye(4)
                    target[:3, :3] = cv2.Rodrigues(rv)[0]
                    target[:3, 3] = tv.ravel()
                    robot.append(t); targets.append(target); stamps.append(stamp); errors.append(error)
                    np.savez_compressed(session / 'samples.npz', T_base_tool=robot, T_camera_target=targets,
                                        image_stamps=stamps, reprojection_error_px=errors, camera_matrix=k, dist_coeffs=d)
                    # Invalidate the previous solve after adding new data.
                    for name in ('result.json', 'T_base_camera.npy'):
                        (session / name).unlink(missing_ok=True)
                    index = len(robot)
                else:
                    if dep.header.frame_id != color.header.frame_id or (dep.width, dep.height) != (color.width, color.height):
                        raise ValueError('Depth must be aligned to color optical frame')
                    z = bridge.imgmsg_to_cv2(dep, 'passthrough').astype(float)
                    if dep.encoding == '16UC1':
                        z *= .001
                    elif dep.encoding != '32FC1':
                        raise ValueError('Depth encoding must be 16UC1 (mm) or 32FC1 (m)')
                    # Polygon mask excludes background; rectangle ROI is insufficient on a table.
                    vertices = []
                    def click(event, x, y, flags, param):
                        if event == cv2.EVENT_LBUTTONDOWN:
                            vertices.append((x, y))
                    cv2.namedWindow('mask'); cv2.setMouseCallback('mask', click)
                    print('Click polygon around visible base/shoulder surface; Enter finish, Esc cancel.')
                    while True:
                        preview = image.copy()
                        if vertices:
                            cv2.polylines(preview, [np.array(vertices, np.int32)], True, (0, 255, 0), 2)
                        cv2.imshow('mask', preview)
                        pressed = cv2.waitKey(20) & 255
                        if pressed in (13, 27):
                            break
                    cv2.destroyWindow('mask')
                    if pressed == 27 or len(vertices) < 3:
                        raise ValueError('Mask cancelled')
                    mask = np.zeros(z.shape, np.uint8)
                    cv2.fillPoly(mask, [np.array(vertices, np.int32)], 1)
                    yy, xx = np.where((mask > 0) & np.isfinite(z) & (z > .1) & (z < 3))
                    if len(xx) < 100:
                        raise ValueError('Too few valid depth points')
                    pixels = np.column_stack([xx, yy]).astype(np.float64).reshape(-1, 1, 2)
                    normalized = cv2.undistortPoints(pixels, k, d).reshape(-1, 2)
                    zz = z[yy, xx]
                    cloud = np.column_stack([normalized * zz[:, None], zz])
                    index = len(list(session.glob('scan_*.npz'))) + 1
                    np.savez_compressed(session / f'scan_{index:03d}.npz', points_camera_m=cloud,
                        model_points_base_m=model, T_base_tool=t, camera_matrix=k, dist_coeffs=d,
                        image_stamp=stamp, base_frame=a.base, camera_frame=color.header.frame_id,
                        model_path=str(a.model.resolve()), mask=mask, depth_m=z, capture_version=3)
                cv2.imwrite(str(session / f'color_{index:03d}.png'), image)
                last_stamp = stamp
                print(f'Saved sample {index}: {session}')
            except Exception as e:
                print('Capture rejected:', e)
    finally:
        node.destroy_node(); rclpy.shutdown(); cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
