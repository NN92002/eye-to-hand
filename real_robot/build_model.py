"""Sample visible robot CAD mesh surfaces using calibrated ROS TF."""
import argparse
from pathlib import Path
import time
import numpy as np


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mesh', action='append', nargs=2, metavar=('LINK_FRAME', 'MESH_PATH'), required=True,
                   help='Repeat for base/shoulder; mesh coordinates must match link frame')
    p.add_argument('--base', default='base')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--count', type=int, default=40000)
    a = p.parse_args()
    import trimesh
    import rclpy
    from rclpy.time import Time
    from tf2_ros import Buffer, TransformListener
    from capture import validate_transform
    rclpy.init()
    node = rclpy.create_node('real_eye_to_hand_cad')
    buffer = Buffer()
    listener = TransformListener(buffer, node)
    points = []
    try:
        deadline = time.monotonic()+10
        while not all(buffer.can_transform(a.base, frame, Time()) for frame, _ in a.mesh):
            rclpy.spin_once(node, timeout_sec=.1)
            if time.monotonic() > deadline:
                raise RuntimeError('CAD link TF unavailable')
        for frame, filename in a.mesh:
            mesh = trimesh.load(filename, force='mesh')
            if mesh.is_empty:
                raise ValueError('Empty mesh: '+filename)
            tf = buffer.lookup_transform(a.base, frame, Time())
            q = tf.transform.rotation
            transform = trimesh.transformations.quaternion_matrix([q.w, q.x, q.y, q.z])
            tr = tf.transform.translation
            transform[:3, 3] = [tr.x, tr.y, tr.z]
            validate_transform(transform)
            sample, _ = trimesh.sample.sample_surface(mesh, max(100, a.count//len(a.mesh)))
            points.append(sample @ transform[:3, :3].T + transform[:3, 3])
        a.output.parent.mkdir(parents=True, exist_ok=True)
        np.save(a.output, np.concatenate(points))
        print('Saved CAD in metres, frame', a.base, 'to', a.output)
    finally:
        node.destroy_node(); rclpy.shutdown()


if __name__ == '__main__':
    main()
