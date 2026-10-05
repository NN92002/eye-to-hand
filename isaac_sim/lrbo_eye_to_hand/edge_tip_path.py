"""Pixel/depth geometry, closed-gripper tip model, preview and slow Isaac execution.

All math uses column vectors. Saved T_base_camera maps ROS optical -> the
calibration robot-root base frame; USD rendering / Lula targets use world frame.
Isaac imports are delayed so geometry and motion guards can be tested offline.
"""
from __future__ import annotations

import copy
import json
from datetime import datetime
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

from lrbo_core import validate_rigid_transform

ARM_JOINTS = ('shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint')
TCP_FRAME = 'tool0'
GRIPPER_BASE_FRAME = 'robotiq_robotiq_140_base_link'
PAD_FRAMES = ('robotiq_left_inner_finger_pad', 'robotiq_right_inner_finger_pad')
CLOSED_GRIPPER_RAD = .7


def finite(value, name):
    value = np.asarray(value, dtype=float)
    if not np.isfinite(value).all():
        raise ValueError(f'{name} contains NaN/Inf')
    return value


def unit(value, name='vector'):
    value = finite(value, name)
    length = np.linalg.norm(value)
    if length < 1e-8:
        raise ValueError(f'{name} has zero length')
    return value / length


def rotation_axis_angle(axis, angle):
    x, y, z = unit(axis)
    cross = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * cross + (1 - np.cos(angle)) * (cross @ cross)


def quaternion(rotation):
    """Proper rotation -> normalized Isaac wxyz quaternion, including 180 degrees."""
    rotation = finite(rotation, 'rotation')
    if rotation.shape != (3, 3) or not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or np.linalg.det(rotation) < .999:
        raise ValueError('Expected a proper 3x3 rotation')
    # Davenport eigenvector form avoids zero denominators near pi.
    r = rotation
    k = np.array([[r[0,0]-r[1,1]-r[2,2], r[0,1]+r[1,0], r[0,2]+r[2,0], r[2,1]-r[1,2]],
                  [r[0,1]+r[1,0], r[1,1]-r[0,0]-r[2,2], r[1,2]+r[2,1], r[0,2]-r[2,0]],
                  [r[0,2]+r[2,0], r[1,2]+r[2,1], r[2,2]-r[0,0]-r[1,1], r[1,0]-r[0,1]],
                  [r[2,1]-r[1,2], r[0,2]-r[2,0], r[1,0]-r[0,1], np.trace(r)]]) / 3
    _, vectors = np.linalg.eigh(k)
    xyzw = vectors[:, -1]
    q = xyzw[[3, 0, 1, 2]]
    return q if q[0] >= 0 else -q


def transform_points(transform, points):
    transform = finite(transform, 'transform')
    if transform.shape != (4, 4) or not np.allclose(transform[3], [0,0,0,1]):
        raise ValueError('Expected an affine 4x4 transform')
    points = finite(points, 'points')
    return points @ transform[:3, :3].T + transform[:3, 3]


def back_project(pixel, depth_m, intrinsics):
    k = finite(intrinsics, 'camera intrinsics')
    if k.shape != (3, 3) or k[0, 0] <= 0 or k[1, 1] <= 0:
        raise ValueError('Invalid camera intrinsics')
    z = float(depth_m) if depth_m is not None else np.nan
    if not np.isfinite(z) or z <= 0:
        raise ValueError('Invalid endpoint median depth')
    u, v = finite(pixel, 'pixel')
    return np.array([(u - k[0, 2]) * z / k[0, 0],
                     (v - k[1, 2]) * z / k[1, 1], z])


def linear_path(first, second, step):
    first, second = finite(first, 'path start'), finite(second, 'path end')
    if step <= 0:
        raise ValueError('Waypoint spacing must be positive')
    count = max(2, int(np.ceil(np.linalg.norm(second - first) / step)) + 1)
    return first + np.linspace(0, 1, count)[:, None] * (second - first)


def tool_orientation(edge, normal, local_axis, local_roll_axis, tilt_deg, toward=None):
    """Map the actual local tool axis to a fixed downward tilted direction.

    The actual local roll reference is mapped along the edge (tool axis is
    perpendicular to the full edge when edge is horizontal; XY always orthogonal).
    """
    normal = unit(normal, 'table normal')
    edge_direction = unit(edge, '3D edge')
    planar = unit(edge_direction - normal * np.dot(edge_direction, normal), 'edge XY projection')
    perpendicular = unit(np.cross(normal, planar))
    if toward is not None and np.dot(perpendicular, toward) < 0:
        perpendicular *= -1
    if not 0 < tilt_deg < 90:
        raise ValueError('Tool tilt must be strictly between 0 and 90 degrees')
    theta = np.radians(tilt_deg)
    direction = np.cos(theta) * perpendicular - np.sin(theta) * normal
    axis = unit(local_axis, 'actual tool axis')
    local_x = unit(local_roll_axis - axis * np.dot(local_roll_axis, axis), 'tool roll reference')
    local_basis = np.column_stack((local_x, np.cross(axis, local_x), axis))
    world_x = planar  # perpendicular to tilted tool direction
    target_basis = np.column_stack((world_x, np.cross(direction, world_x), direction))
    rotation = target_basis @ local_basis.T
    return rotation, edge_direction, direction


def assert_bounds(points, lower, upper, name):
    points = finite(points, name)
    if np.any(points < np.asarray(lower) - 1e-8) or np.any(points > np.asarray(upper) + 1e-8):
        raise ValueError(f'{name} is outside workspace bounds')


def fixed_tip_vertical_turn(rotation, offset, tip, edge_direction, normal, tool_axis):
    """Tilt about the horizontal edge direction, keeping physical tip fixed."""
    normal = unit(normal)
    axis = unit(edge_direction - normal*np.dot(edge_direction, normal))
    forward = rotation @ unit(tool_axis)
    angle = np.arctan2(np.dot(axis, np.cross(forward, -normal)), np.dot(forward, -normal))
    rotations = np.array([rotation_axis_angle(axis, t*angle) @ rotation
                          for t in np.linspace(0, 1, max(2, int(np.ceil(abs(angle)/np.radians(.5)))+1))])
    tcp = np.array([tip-r@offset for r in rotations])
    if np.linalg.norm(rotations[-1]@unit(tool_axis)+normal) > 1e-6:
        raise ValueError('Cannot turn tool vertical about edge direction')
    return rotations, tcp, angle


def build_edge_plan(edge, depth_info, intrinsics, base_camera, world_base,
                    d_tcp_tip, tool_axis, roll_axis, table_normal, clearance,
                    tilt_deg, spacing, toward, tip_bounds, tcp_bounds,
                    depth_consistency_m=.025):
    if edge is None or depth_info is None:
        raise ValueError('Press DETECT ONCE and obtain a valid edge/depth first')
    if clearance < 0:
        raise ValueError('EDGE_CLEARANCE must be nonnegative')
    if depth_info['valid_count'] < .8 * depth_info['sample_count']:
        raise ValueError(f"Invalid edge depth: {depth_info.get('status', 'need 80% valid samples')}")
    base_camera = validate_rigid_transform(base_camera)
    world_base = validate_rigid_transform(world_base)
    camera = np.stack([back_project(edge['p1'], depth_info['p1_surface_m'], intrinsics),
                       back_project(edge['p2'], depth_info['p2_surface_m'], intrinsics)])
    base = transform_points(base_camera, camera)
    length = np.linalg.norm(base[1] - base[0])
    if not .005 <= length <= .25:
        raise ValueError(f'Implausible 3D edge length: {length:.4f} m')
    # A single 3D segment must explain the nearby depth samples; reject table
    # transitions / occlusion rather than generate a line across inconsistent Z.
    difference = camera[1] - camera[0]
    for sample in depth_info['samples']:
        if sample['depth_m'] is None:
            continue
        point = back_project(sample['edge_uv'], sample['depth_m'], intrinsics)
        t = np.clip(np.dot(point - camera[0], difference) / np.dot(difference, difference), 0, 1)
        if np.linalg.norm(point - (camera[0] + t * difference)) > depth_consistency_m:
            raise ValueError('Edge depth profile is inconsistent with a straight 3D edge')
    rotation, direction, tool_direction = tool_orientation(
        base[1] - base[0], table_normal, tool_axis, roll_axis, tilt_deg, toward)
    normal = unit(table_normal)
    tip = linear_path(base[0] + clearance * normal, base[1] + clearance * normal, spacing)
    offset = finite(d_tcp_tip, 'TCP-to-tip offset')
    tcp = tip - rotation @ offset
    world_tip = transform_points(world_base, tip)
    world_tcp = transform_points(world_base, tcp)
    assert_bounds(world_tip, *tip_bounds, 'Desired tip path (world)')
    assert_bounds(world_tcp, *tcp_bounds, 'TCP path (world)')
    turn_rotations, turn_tcp, turn_angle = fixed_tip_vertical_turn(
        rotation, offset, tip[-1], direction, normal, tool_axis)
    world_turn_tcp = transform_points(world_base, turn_tcp)
    assert_bounds(world_turn_tcp, *tcp_bounds, 'Final rotation TCP path (world)')
    return {'pixels': (edge['p1'], edge['p2']), 'camera': camera, 'base': base,
            'base_camera': base_camera.copy(),
            'rotation': rotation, 'quaternion_wxyz': quaternion(rotation),
            'edge_direction': direction, 'tool_direction': tool_direction,
            'offset': offset, 'tip': tip, 'tcp': tcp, 'world_base': world_base.copy(),
            'world_tip': world_tip, 'world_tcp': world_tcp,
            'tip_bounds': tip_bounds, 'tcp_bounds': tcp_bounds,
            'turn_rotations': turn_rotations, 'world_turn_tcp': world_turn_tcp,
            'turn_angle_rad': turn_angle}


class URDFModel:
    """Read exact scene URDF, including fixed frame rotations and gripper geometry."""
    def __init__(self, path):
        self.path = Path(path).resolve()
        self.root = ET.parse(path).getroot()
        self.links = {node.get('name'): node for node in self.root.findall('link')}
        self.joints = {node.find('child').get('link'): node for node in self.root.findall('joint')}

    @staticmethod
    def origin(node):
        origin = node.find('origin')
        matrix = np.eye(4)
        if origin is not None:
            matrix[:3, 3] = np.fromstring(origin.get('xyz', '0 0 0'), sep=' ')
            roll, pitch, yaw = np.fromstring(origin.get('rpy', '0 0 0'), sep=' ')
            matrix[:3, :3] = (rotation_axis_angle([0,0,1], yaw) @
                             rotation_axis_angle([0,1,0], pitch) @ rotation_axis_angle([1,0,0], roll))
        return matrix

    def pose(self, frame, values):
        if frame not in self.joints:
            return np.eye(4)
        joint = self.joints[frame]
        matrix = self.pose(joint.find('parent').get('link'), values) @ self.origin(joint)
        if joint.get('type') in ('revolute', 'continuous'):
            axis_node = joint.find('axis')
            axis = np.fromstring(axis_node.get('xyz', '1 0 0'), sep=' ')
            mimic = joint.find('mimic')
            angle = values.get(joint.get('name'))
            if angle is None and mimic is not None:
                angle = values.get(mimic.get('joint'), 0) * float(mimic.get('multiplier', 1)) + float(mimic.get('offset', 0))
            matrix[:3, :3] = matrix[:3, :3] @ rotation_axis_angle(axis, angle or 0)
        return matrix

    def closed_tip(self, multipliers):
        values = {name: CLOSED_GRIPPER_RAD * ratio for name, ratio in multipliers.items()}
        tcp = self.pose(TCP_FRAME, values)
        base = np.linalg.inv(tcp) @ self.pose(GRIPPER_BASE_FRAME, values)
        axis = unit(base[:3, :3] @ [0, 0, 1], 'URDF gripper extension axis')
        roll = base[:3, :3] @ [1, 0, 0]
        points, local_points = [], {}
        for name in PAD_FRAMES:
            tcp_pad = np.linalg.inv(tcp) @ self.pose(name, values)
            visual = self.links[name].find('visual')
            box = visual.find('geometry/box')
            if box is None:
                raise ValueError('Finger pad box missing; explicitly configure the tip offset')
            size = np.fromstring(box.get('size'), sep=' ')
            visual_pose = self.origin(visual)
            axis_pad = (tcp_pad @ visual_pose)[:3, :3].T @ axis
            distal_axis = int(np.argmax(np.abs(axis_pad)))
            local = np.zeros(3)
            local[distal_axis] = np.sign(axis_pad[distal_axis]) * size[distal_axis] / 2
            local = transform_points(visual_pose, local)
            local_points[name] = local
            points.append(transform_points(tcp_pad, local))
        return np.mean(points, axis=0), axis, roll, local_points

    def collision_vertices(self):
        """Read exact URDF collision origins/scale; independent of USD pose caches."""
        import struct
        workspace = self.path.parent.parent
        roots = {'ur_description': workspace / 'src/universal_robot',
                 'robotiq_description': workspace / 'src/robotiq_ros/grippers/robotiq_description'}
        result = []
        for name, link in self.links.items():
            pieces = []
            for collision in link.findall('collision'):
                geometry = collision.find('geometry')
                box, mesh = geometry.find('box'), geometry.find('mesh')
                if box is not None:
                    half = np.fromstring(box.get('size'), sep=' ') / 2
                    vertices = np.array([[x,y,z] for x in (-half[0],half[0])
                                         for y in (-half[1],half[1]) for z in (-half[2],half[2])])
                elif mesh is not None:
                    filename = mesh.get('filename')
                    if filename.startswith('package://'):
                        package, relative = filename[len('package://'):].split('/',1)
                        filename = roots[package] / relative
                    else:
                        filename = self.path.parent / filename
                    data = Path(filename).read_bytes()
                    count = struct.unpack_from('<I', data, 80)[0] if len(data)>=84 else 0
                    if len(data) == 84 + 50*count:
                        dtype = np.dtype([('normal','<f4',(3,)),('vertices','<f4',(3,3)),('attribute','<u2')])
                        vertices = np.frombuffer(data,dtype=dtype,count=count,offset=84)['vertices'].reshape(-1,3)
                    else:
                        vertices = np.array([list(map(float,line.split()[1:])) for line in
                                             data.decode('ascii').splitlines() if line.strip().startswith('vertex ')])
                    if not len(vertices):
                        raise ValueError(f'Empty collision mesh: {filename}')
                    vertices = vertices * np.fromstring(mesh.get('scale','1 1 1'),sep=' ')
                else:
                    raise ValueError(f'Unsupported URDF collision geometry: {name}')
                pieces.append(transform_points(self.origin(collision),vertices))
            if pieces:
                result.append((name,np.unique(np.concatenate(pieces),axis=0)))
        return result

    def write_lula_arm_model(self, directory):
        """Reduce exact scene model to root -> tool0 chain; don't alter scene URDF."""
        joints, frames = [], [TCP_FRAME]
        child = TCP_FRAME
        while child in self.joints:
            joint = self.joints[child]
            joints.append(copy.deepcopy(joint))
            child = joint.find('parent').get('link')
            frames.append(child)
        root = ET.Element('robot', name='existing_scene_ur5e_arm')
        for frame in reversed(frames):
            ET.SubElement(root, 'link', name=frame)
        for joint in reversed(joints):
            root.append(joint)
        urdf = Path(directory) / 'scene_arm.urdf'
        ET.ElementTree(root).write(urdf, encoding='unicode')
        descriptor = Path(directory) / 'scene_arm.yaml'
        descriptor.write_text('api_version: 1.0\ncspace:\n' +
                              ''.join(f'  - {name}\n' for name in ARM_JOINTS) +
                              'default_q: [0, -1.57, 1.57, -1.57, -1.57, 0]\n'
                              'cspace_to_urdf_rules: []\ncollision_spheres: []\n')
        return descriptor, urdf


def draw_plan(scene, stage, plan):
    """Visual-only USD primitives: never add colliders or modify scene assets."""
    from pxr import UsdGeom, Gf
    root = '/CubeEdgePathPreview'
    stage.RemovePrim(root)
    UsdGeom.Xform.Define(stage, root)
    world_edge = transform_points(plan['world_base'], plan['base'])
    for name, points, color in (
            ('DetectedEdge', world_edge, (1,0,.2)),
            ('DesiredTipPath', plan['world_tip'], (0,1,1)),
            ('CalculatedTCPPath', plan['world_tcp'], (1,.55,0)),
            ('FinalRotationTCP', plan['world_turn_tcp'], (.2,.6,1))):
        curve = UsdGeom.BasisCurves.Define(stage, root + '/' + name)
        curve.CreateTypeAttr('linear')
        curve.CreateWrapAttr('nonperiodic')
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr([Gf.Vec3f(*map(float, point)) for point in points])
        curve.CreateWidthsAttr([.003])
        curve.SetWidthsInterpolation('constant')
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
    for index, point in enumerate(world_edge):
        sphere = UsdGeom.Sphere.Define(stage, f'{root}/P{index+1}_base')
        sphere.CreateRadiusAttr(.006)
        sphere.CreateDisplayColorAttr([Gf.Vec3f(1,0,.2)])
        UsdGeom.XformCommonAPI(sphere.GetPrim()).SetTranslate(Gf.Vec3d(*map(float, point)))


class SlowEdgeExecutor:
    """Preflight all IK samples then drive existing Isaac articulation at low speed.

    Lula IK is not a collision planner. Geometry is checked for tabletop/cube
    intersection, joint limits/jumps, and FK/scene agreement before any motion.
    """
    def __init__(self, scene, stage, robot, world, robot_base_path, tool_path, model, settings, status):
        self.scene, self.stage, self.robot, self.world = scene, stage, robot, world
        self.base_path, self.tool_path = robot_base_path, tool_path
        self.model, self.settings, self.status = model, settings, status
        self.active = False
        self.plan = None
        self.solver = None
        self._temp = None
        self.samples = []
        self.elapsed = 0.0
        self.last_time = None
        self.trace_file = None
        self.last_trace_time = -np.inf
        self.gripper_indices = np.array([list(robot.dof_names).index(name)
                                        for name in scene.ROBOTIQ_JOINT_MULTIPLIERS], dtype=int)
        self.closed_targets = np.array([CLOSED_GRIPPER_RAD * ratio
                                       for ratio in scene.ROBOTIQ_JOINT_MULTIPLIERS.values()])

    def _matrix(self, path):
        return self.scene.get_prim_world_matrix(self.stage, path)

    def _joints(self):
        return finite(self.robot.get_joint_positions(), 'robot joints')

    def _solver(self):
        if self.solver is None:
            from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver
            self._temp = tempfile.TemporaryDirectory(prefix='cube_edge_lula_')
            description, urdf = self.model.write_lula_arm_model(self._temp.name)
            self.solver = LulaKinematicsSolver(str(description), str(urdf))
            self.arm_indices = np.array([list(self.robot.dof_names).index(name)
                                         for name in self.solver.get_joint_names()], dtype=int)
        base = self._matrix(self.base_path)
        self.solver.set_robot_base_pose(base[:3,3], quaternion(base[:3,:3]))
        q = self._joints()[self.arm_indices]
        position, rotation = self.solver.compute_forward_kinematics(TCP_FRAME, q)
        measured = self._matrix(self.tool_path)
        if np.linalg.norm(position - measured[:3,3]) > .002 or np.linalg.norm(rotation - measured[:3,:3]) > .02:
            raise ValueError('URDF FK and actual USD TCP disagree; execution rejected (frame/joint convention)')
        return q

    def start(self, plan):
        if not self.settings.EXECUTE_ROBOT:
            self.status('PREVIEW ONLY: EXECUTE_ROBOT=False; no robot/gripper commands')
            return
        if self.active:
            raise ValueError('Already executing')
        self._solver()  # validate native model before closing gripper
        controller = self.robot.get_articulation_controller()
        kps, kds = controller.get_gains()
        kps, kds = np.asarray(kps).reshape(-1).copy(), np.asarray(kds).reshape(-1).copy()
        kps[self.arm_indices] = self.settings.PATH_ARM_STIFFNESS
        kds[self.arm_indices] = self.settings.PATH_ARM_DAMPING
        controller.set_gains(kps=kps, kds=kds, save_to_usd=False)
        self.plan = plan
        trace_dir = Path(__file__).parent / 'comparisons' / 'executions'
        trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_file = trace_dir / f"execution_{datetime.now():%Y%m%d_%H%M%S_%f}.jsonl"
        self.last_trace_time = -np.inf
        print(f'Execution trace: {self.trace_file}', flush=True)
        self._apply(self._joints()[self.arm_indices], self.arm_indices)
        self.active = True
        self.phase = 'closing'
        self.elapsed = 0
        self.last_time = float(self.world.current_time)
        self._apply(self.closed_targets, self.gripper_indices)
        self.status('Closing gripper before preflight; arm holds current pose')

    def _apply(self, positions, indices):
        self.robot.get_articulation_controller().apply_action(
            self.scene.ArticulationAction(joint_positions=np.asarray(positions, dtype=np.float32),
                                          joint_indices=np.asarray(indices, dtype=np.int32)))

    def stop(self, reason='Stopped'):
        if self.active:
            current = self._joints()
            indices = np.concatenate((self.arm_indices, self.gripper_indices))
            self._apply(current[indices], indices)
        self.active = False
        self.status(reason)
        if self.trace_file is not None:
            with self.trace_file.open('a') as stream:
                stream.write(json.dumps({'event': 'stop', 'reason': reason, 'elapsed_s': self.elapsed})+'\n')
        print(reason, flush=True)

    def _geometry_check(self, q, world_tcp):
        """Check sampled arm/link geometry vs workspace obstacles at a proposed pose."""
        values = dict(zip(self.solver.get_joint_names(), q))
        values.update(dict(zip(self.scene.ROBOTIQ_JOINT_MULTIPLIERS, self.closed_targets)))
        world_base = self.plan['world_base']
        tabletop = self.scene.TABLE_HEIGHT + self.scene.TABLE_THICKNESS
        center = self.scene.WORKSPACE_CENTER[:2]
        half = np.array([self.scene.TABLE_WIDTH, self.scene.TABLE_DEPTH]) / 2
        if not hasattr(self, '_geometry'):
            self._geometry = self.model.collision_vertices()
            covered = {link for link, _ in self._geometry}
            if not set(PAD_FRAMES).issubset(covered):
                raise ValueError('Finger pad collision geometry missing')
        cube_inverse = np.linalg.inv(self._matrix('/GreenCube'))
        cube_half = self.settings.CUBE_SIZE_M / 2
        for link, points in self._geometry:
            if link in ('world', 'base_link', 'base_link_inertia', 'base'):
                continue
            world_points = transform_points(world_base @ self.model.pose(link, values), points)
            on_table = np.all(np.abs(world_points[:,:2] - center) <= half, axis=1)
            if np.any(on_table & (world_points[:,2] < tabletop + .001)):
                point = world_points[np.flatnonzero(on_table & (world_points[:,2] < tabletop + .001))[0]]
                raise ValueError(f'Predicted {link} intersects tabletop at world {point.round(4).tolist()}')
            cube_points = transform_points(cube_inverse, world_points)
            if np.any(np.all(np.abs(cube_points) < cube_half + .001, axis=1)):
                raise ValueError(f'Predicted {link} intersects green cube')

    def _preflight(self):
        failures = []
        for extra_height in (0.0, 0.08, 0.16):
            try:
                self._preflight_candidate(extra_height)
                return
            except ValueError as error:
                failures.append(f'+{extra_height*1000:.0f} mm: {error}')
                print(f'Approach candidate rejected: {failures[-1]}', flush=True)
        raise ValueError('No validated approach path; ' + ' | '.join(failures))

    def _preflight_candidate(self, extra_height):
        q = self._solver()
        current = self._matrix(self.tool_path)
        base = self.plan['world_base']
        rotation_world = base[:3,:3] @ self.plan['rotation']
        tip = self.plan['world_tip']
        approach_clearance = self.settings.PRE_START_CLEARANCE + extra_height
        pre_tip = tip[0] + base[:3,:3] @ np.array([0,0,approach_clearance])
        pre_tcp = pre_tip - rotation_world @ self.plan['offset']
        assert_bounds(np.array([pre_tip]), *self.plan['tip_bounds'], 'Pre-start tip')
        assert_bounds(np.array([pre_tcp]), *self.plan['tcp_bounds'], 'Pre-start TCP')
        # Raise current TCP before reorientation/travel, then approach pre-start.
        lift = current[:3,3].copy()
        lift[2] = max(lift[2], pre_tcp[2]) + self.settings.PRE_START_CLEARANCE
        segments = [(current[:3,3], lift, current[:3,:3], current[:3,:3], 'lift'),
                    (lift, pre_tcp, current[:3,:3], rotation_world, 'pre-start'),
                    (pre_tcp, self.plan['world_tcp'][0], rotation_world, rotation_world, 'edge-start')]
        segments.extend((a,b,rotation_world,rotation_world,'edge')
                        for a,b in zip(self.plan['world_tcp'][:-1], self.plan['world_tcp'][1:]))
        turn_rotations = np.array([base[:3,:3] @ r for r in self.plan['turn_rotations']])
        turn_tcp = self.plan['world_turn_tcp']
        segments.extend((a,b,ra,rb,'rotate-end') for a,b,ra,rb in
                        zip(turn_tcp[:-1],turn_tcp[1:],turn_rotations[:-1],turn_rotations[1:]))
        # Incremental orientation interpolation avoids a sudden reorientation.
        samples, times, phases = [q.copy()], [0.0], ['current']
        approach_tip = [current[:3,3] + current[:3,:3] @ self.plan['offset']]
        approach_tcp = [current[:3,3].copy()]
        position = current[:3,3]
        previous_tip = position + current[:3,:3] @ self.plan['offset']
        lower, upper = self.solver.get_cspace_position_limits()
        for a,b,ra,rb,phase in segments:
            relative = ra.T @ rb
            angle = np.arccos(np.clip((np.trace(relative)-1)/2, -1, 1))
            if angle > 1e-8:
                # Eigenvector corresponding to eigenvalue 1 works at 180 degrees.
                vals, vectors = np.linalg.eig(relative)
                axis = np.real(vectors[:, np.argmin(np.abs(vals-1))]); axis = unit(axis)
                skew = np.array([relative[2,1]-relative[1,2],relative[0,2]-relative[2,0],relative[1,0]-relative[0,1]])
                if np.dot(axis, skew) < 0: axis *= -1
            else:
                axis = np.array([1.,0,0])
            count = max(1, int(np.ceil(np.linalg.norm(b-a)/self.settings.WAYPOINT_SPACING)),
                        int(np.ceil(angle / np.radians(2))))
            for fraction in np.linspace(0,1,count+1)[1:]:
                point = a + fraction*(b-a)
                rotation = ra @ rotation_axis_angle(axis, fraction*angle)
                if phase == 'rotate-end':
                    point = tip[-1] - rotation @ self.plan['offset']
                assert_bounds(point, *self.plan['tcp_bounds'], f'{phase} TCP')
                solution, success = self.solver.compute_inverse_kinematics(
                    TCP_FRAME, point, quaternion(rotation), warm_start=q,
                    position_tolerance=.0005, orientation_tolerance=np.radians(.25))
                if not success:
                    raise ValueError(f'IK failed during {phase}; no arm movement performed')
                solution = finite(solution, 'IK solution')
                # Choose closest equivalent joint representation within limits.
                candidates = solution[:,None] + 2*np.pi*np.arange(-2,3)
                distances = np.abs(candidates-q[:,None])
                distances[(candidates<lower[:,None]) | (candidates>upper[:,None])] = np.inf
                choice = np.argmin(distances,axis=1)
                solution = candidates[np.arange(len(q)),choice]
                if np.any(~np.isfinite(distances[np.arange(len(q)),choice])):
                    raise ValueError('IK joint limit violation')
                jump = np.max(np.abs(solution-q))
                if jump > self.settings.MAX_JOINT_WAYPOINT_JUMP:
                    raise ValueError(f'IK branch jump {np.degrees(jump):.1f} deg during {phase}')
                if np.linalg.norm(point-position) > self.settings.MAX_CARTESIAN_WAYPOINT_JUMP:
                    raise ValueError('Cartesian waypoint jump')
                # Screen interpolated joint poses, not just IK endpoints. Ensure
                # the closed physical tip stays close to its Cartesian segment.
                desired_tip = point + rotation @ self.plan['offset']
                if phase not in ('edge', 'rotate-end'):
                    approach_tip.append(desired_tip.copy())
                    approach_tcp.append(point.copy())
                for blend in (.25,.5,.75,1.):
                    intermediate = q + blend*(solution-q)
                    fk_position, fk_rotation = self.solver.compute_forward_kinematics(TCP_FRAME, intermediate)
                    actual_tip = fk_position + fk_rotation @ self.plan['offset']
                    expected_tip = previous_tip + blend*(desired_tip-previous_tip)
                    if np.linalg.norm(actual_tip-expected_tip) > .003:
                        raise ValueError('Interpolated IK segment deviates from tip path (>3 mm)')
                    if phase in ('edge-start','edge','rotate-end'):
                        assert_bounds(actual_tip,*self.plan['tip_bounds'],'Interpolated tip')
                    else:
                        assert_bounds(actual_tip,*self.plan['tcp_bounds'],'Approach tip')
                    try:
                        self._geometry_check(intermediate, fk_position)
                    except ValueError as error:
                        if self.trace_file is not None:
                            with self.trace_file.open('a') as stream:
                                stream.write(json.dumps({'event': 'preflight_rejected', 'phase': phase,
                                    'fraction': float(fraction), 'joint_blend': blend,
                                    'joints_rad': intermediate.tolist(), 'reason': str(error)})+'\n')
                        raise ValueError(f'{phase} preflight: {error}') from error
                dt = max(self.settings.MIN_WAYPOINT_DURATION_S,
                         1.875*np.linalg.norm(desired_tip-previous_tip)/self.settings.TIP_SPEED_M_S,
                         1.875*jump/self.settings.JOINT_SPEED_RAD_S,
                         np.sqrt(6*jump/self.settings.JOINT_ACCEL_RAD_S2))
                times.append(times[-1]+dt); samples.append(solution.copy()); phases.append(phase)
                q, position, previous_tip = solution, point, desired_tip
        self.samples, self.times, self.phases = np.array(samples), np.array(times), phases
        from pxr import UsdGeom, Gf
        for name, points, color in (('ApproachTip', approach_tip, (.8,.3,1)),
                                    ('ApproachTCP', approach_tcp, (1,1,0))):
            curve = UsdGeom.BasisCurves.Define(self.stage, '/CubeEdgePathPreview/'+name)
            curve.CreateTypeAttr('linear')
            curve.CreateWrapAttr('nonperiodic')
            curve.CreateCurveVertexCountsAttr([len(points)])
            curve.CreatePointsAttr([Gf.Vec3f(*map(float,p)) for p in points])
            curve.CreateWidthsAttr([.002])
            curve.SetWidthsInterpolation('constant')
            curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        self.elapsed = 0
        self.phase = 'moving'
        self.status(f'Preflight OK: approach clearance {approach_clearance*1000:.0f} mm; '
                    f'{len(samples)} IK samples; estimated {times[-1]:.1f} s')

    def tick(self):
        if not self.active:
            return
        try:
            now = float(self.world.current_time)
            dt = now - self.last_time
            if dt < 0:
                raise ValueError('Simulation reset during execution')
            self.last_time = now
            self.elapsed += dt
            joints = self._joints()
            if not np.allclose(self._matrix(self.base_path), self.plan['world_base'], atol=1e-5):
                raise ValueError('Robot base moved after planning')
            if self.phase == 'closing':
                if np.max(np.abs(joints[self.gripper_indices]-self.closed_targets)) <= .015:
                    # Compare actual tip in tool0 to the URDF closed-tip offset.
                    local_points = self.model.closed_tip(self.scene.ROBOTIQ_JOINT_MULTIPLIERS)[3]
                    actual = []
                    for name, local in local_points.items():
                        matches = [str(p.GetPath()) for p in self.stage.Traverse()
                                   if p.GetName()==name and str(p.GetPath()).startswith(self.base_path+'/')]
                        if len(matches)!=1: raise ValueError(f'Cannot resolve actual fingertip {name}')
                        actual.append(transform_points(np.linalg.inv(self._matrix(self.tool_path)) @ self._matrix(matches[0]), local))
                    if np.linalg.norm(np.mean(actual,axis=0)-self.plan['offset']) > .003:
                        raise ValueError('Actual closed-gripper tip offset differs from model (>3 mm)')
                    self._preflight()
                elif self.elapsed > 8:
                    raise ValueError('Gripper did not reach closed configuration')
                return
            if not np.allclose(self._matrix('/GreenCube'), self.plan['cube_world'], atol=1e-5, rtol=0):
                raise ValueError('Cube moved during execution')
            if np.max(np.abs(joints[self.gripper_indices]-self.closed_targets)) > .025:
                raise ValueError('Gripper moved; TCP-tip offset invalid')
            index = min(len(self.times)-2, int(np.searchsorted(self.times,self.elapsed,side='right'))-1)
            t = np.clip((self.elapsed-self.times[index])/(self.times[index+1]-self.times[index]),0,1)
            smooth = 10*t**3 - 15*t**4 + 6*t**5  # zero velocity/accel at segment endpoints
            target = self.samples[index] + smooth*(self.samples[index+1]-self.samples[index])
            fk_position, fk_rotation = self.solver.compute_forward_kinematics(TCP_FRAME, target)
            commanded_tip = fk_position + fk_rotation @ self.plan['offset']
            actual_tcp = self._matrix(self.tool_path)
            actual_tip = actual_tcp[:3,3] + actual_tcp[:3,:3] @ self.plan['offset']
            tip_error = float(np.linalg.norm(actual_tip-commanded_tip))
            phase = self.phases[index+1]
            if self.elapsed-self.last_trace_time >= .1:
                self.last_trace_time = self.elapsed
                record = {'elapsed_s': self.elapsed, 'phase': phase,
                          'command_tip_world_m': commanded_tip.tolist(),
                          'actual_tip_world_m': actual_tip.tolist(),
                          'target_joints_rad': target.tolist(),
                          'actual_joints_rad': joints[self.arm_indices].tolist(),
                          'tip_tracking_error_mm': tip_error*1000}
                with self.trace_file.open('a') as stream:
                    stream.write(json.dumps(record)+'\n')
                self.status(f'Moving: {phase}; {self.elapsed:.1f}/{self.times[-1]:.1f} s; '
                            f'tip tracking error {tip_error*1000:.2f} mm')
            if np.max(np.abs(joints[self.arm_indices]-target)) > self.settings.MAX_TRACKING_ERROR_RAD:
                raise ValueError('Joint tracking error too large')
            if tip_error > self.settings.MAX_TIP_TRACKING_ERROR_M:
                raise ValueError(f'Tip tracking error {tip_error*1000:.2f} mm during {phase}')
            self._apply(np.concatenate((target, self.closed_targets)),
                        np.concatenate((self.arm_indices, self.gripper_indices)))
            if self.elapsed >= self.times[-1]:
                if np.max(np.abs(joints[self.arm_indices]-self.samples[-1])) < .015:
                    self.active = False
                    actual_tcp = self._matrix(self.tool_path)
                    actual_tip = actual_tcp[:3,3] + actual_tcp[:3,:3] @ self.plan['offset']
                    tip_error = np.linalg.norm(actual_tip-self.plan['world_tip'][-1])
                    message = f'Completed edge path + vertical turn; tip tracking error {tip_error*1000:.2f} mm; holding final pose'
                    self.status(message)
                    print(message, flush=True)
                elif self.elapsed > self.times[-1]+5:
                    raise ValueError('Edge-end settling timeout')
        except Exception as error:
            self.stop(f'Execution stopped: {error}')

    def close(self):
        self.stop('Closed')
        if self._temp is not None:
            self._temp.cleanup()
