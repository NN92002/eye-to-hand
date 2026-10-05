import math

import rclpy
from geometry_msgs.msg import Point, TransformStamped
from rclpy.node import Node
from tf2_ros import StaticTransformBroadcaster
from visualization_msgs.msg import Marker, MarkerArray


def normalize(vector):
    length = math.sqrt(sum(value * value for value in vector))
    if length < 1e-9:
        raise ValueError('Cannot normalize a zero-length vector')
    return tuple(value / length for value in vector)


def cross(left, right):
    return (left[1] * right[2] - left[2] * right[1], left[2] * right[0] - left[0] * right[2], left[0] * right[1] - left[1] * right[0])


def dot(left, right):
    return sum(a * b for a, b in zip(left, right))


def quaternion_from_basis(x_axis, y_axis, z_axis):
    matrix = (x_axis, y_axis, z_axis)
    trace = sum(matrix[index][index] for index in range(3))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return ((y_axis[2] - z_axis[1]) / scale, (z_axis[0] - x_axis[2]) / scale, (x_axis[1] - y_axis[0]) / scale, 0.25 * scale)
    diagonal = [matrix[index][index] for index in range(3)]
    largest = diagonal.index(max(diagonal))
    next_index, last_index = (largest + 1) % 3, (largest + 2) % 3
    scale = math.sqrt(1.0 + diagonal[largest] - diagonal[next_index] - diagonal[last_index]) * 2.0
    quaternion = [0.0, 0.0, 0.0, 0.0]
    quaternion[largest] = 0.25 * scale
    quaternion[3] = (matrix[next_index][last_index] - matrix[last_index][next_index]) / scale
    quaternion[next_index] = (matrix[largest][next_index] + matrix[next_index][largest]) / scale
    quaternion[last_index] = (matrix[largest][last_index] + matrix[last_index][largest]) / scale
    return tuple(quaternion)


class WorkspaceVisualizer(Node):
    def __init__(self):
        super().__init__('workspace_visualizer')
        self._declare_parameters()
        self._read_parameters()
        self.marker_pub = self.create_publisher(MarkerArray, 'workspace_markers', 10)
        self.static_tf = StaticTransformBroadcaster(self)
        self._publish_static_camera_tf()
        self.create_timer(1.0 / self.publish_rate, self._publish_markers)
        self._report_geometry()

    def _declare_parameters(self):
        defaults = {
            'base_frame': 'base_link', 'camera_link_frame': 'camera_link', 'camera_optical_frame': 'camera_color_optical_frame',
            'table_width': 0.70, 'table_depth': 0.70, 'table_height': 0.35,
            'workspace_center_x': 0.45, 'workspace_center_y': 0.0, 'workspace_center_z': 0.35,
            'camera_front_axis_x': -1.0, 'camera_front_axis_y': 0.0,
            'camera_horizontal_distance': 0.70, 'camera_height_above_table': 0.90,
            'camera_hfov': 1.22, 'camera_vfov': 0.76, 'camera_near': 0.10, 'camera_far': 5.0, 'publish_rate': 10.0,
        }
        for name, value in defaults.items():
            self.declare_parameter(name, value)

    def _read_parameters(self):
        get = lambda name: self.get_parameter(name).value
        self.base_frame, self.camera_link_frame, self.camera_optical_frame = get('base_frame'), get('camera_link_frame'), get('camera_optical_frame')
        self.width, self.depth, self.table_height = get('table_width'), get('table_depth'), get('table_height')
        self.center = (get('workspace_center_x'), get('workspace_center_y'), get('workspace_center_z'))
        self.front = normalize((get('camera_front_axis_x'), get('camera_front_axis_y'), 0.0))
        distance = get('camera_horizontal_distance')
        self.camera_position = (self.center[0] + self.front[0] * distance, self.center[1] + self.front[1] * distance, self.table_height + get('camera_height_above_table'))
        self.hfov, self.vfov = get('camera_hfov'), get('camera_vfov')
        self.near, self.far, self.publish_rate = get('camera_near'), get('camera_far'), get('publish_rate')
        self.forward = normalize(tuple(a - b for a, b in zip(self.center, self.camera_position)))
        self.right = normalize(cross(self.forward, (0.0, 0.0, 1.0)))
        self.down = normalize(cross(self.forward, self.right))
        self.quaternion = quaternion_from_basis(self.right, self.down, self.forward)

    def _publish_static_camera_tf(self):
        now = self.get_clock().now().to_msg()
        link = TransformStamped()
        link.header.stamp, link.header.frame_id, link.child_frame_id = now, self.base_frame, self.camera_link_frame
        link.transform.translation.x, link.transform.translation.y, link.transform.translation.z = self.camera_position
        link.transform.rotation.x, link.transform.rotation.y, link.transform.rotation.z, link.transform.rotation.w = self.quaternion
        optical = TransformStamped()
        optical.header.stamp, optical.header.frame_id, optical.child_frame_id = now, self.camera_link_frame, self.camera_optical_frame
        optical.transform.rotation.w = 1.0
        self.static_tf.sendTransform([link, optical])

    def _corners(self):
        half_width, half_depth = self.width / 2.0, self.depth / 2.0
        return [(self.center[0] + dx, self.center[1] + dy, self.table_height) for dx, dy in ((-half_width, -half_depth), (-half_width, half_depth), (half_width, half_depth), (half_width, -half_depth))]

    def _camera_coordinates(self, point, position=None):
        position = position or self.camera_position
        relative = tuple(a - b for a, b in zip(point, position))
        return dot(relative, self.right), dot(relative, self.down), dot(relative, self.forward)

    def _inside_frustum(self, point):
        x, y, z = self._camera_coordinates(point)
        return self.near <= z <= self.far and abs(x) <= z * math.tan(self.hfov / 2.0) and abs(y) <= z * math.tan(self.vfov / 2.0)

    def _report_geometry(self):
        horizontal = math.hypot(self.center[0] - self.camera_position[0], self.center[1] - self.camera_position[1])
        height = self.camera_position[2] - self.table_height
        angle = math.degrees(math.atan2(horizontal, height))
        inside = [self._inside_frustum(corner) for corner in self._corners()]
        self.get_logger().info('workspace %.3f x %.3f m, camera horizontal distance %.3f m, height %.3f m' % (self.width, self.depth, horizontal, height))
        self.get_logger().info('optical axis angle from vertical down: %.2f deg; quaternion=%s' % (angle, self.quaternion))
        self.get_logger().info('workspace corners in D435 frustum: %s' % inside)
        if not all(inside):
            self.get_logger().warn('Workspace is not fully inside the configured FOV. Suggested horizontal distance: %.3f m' % self._suggest_distance())

    def _suggest_distance(self):
        for distance in [index / 100.0 for index in range(5, 1001)]:
            position = (self.center[0] + self.front[0] * distance, self.center[1] + self.front[1] * distance, self.camera_position[2])
            forward = normalize(tuple(a - b for a, b in zip(self.center, position)))
            right = normalize(cross(forward, (0.0, 0.0, 1.0)))
            down = normalize(cross(forward, right))
            valid = True
            for corner in self._corners():
                relative = tuple(a - b for a, b in zip(corner, position))
                x, y, z = dot(relative, right), dot(relative, down), dot(relative, forward)
                valid = valid and self.near <= z <= self.far and abs(x) <= z * math.tan(self.hfov / 2.0) and abs(y) <= z * math.tan(self.vfov / 2.0)
            if valid:
                return distance
        return float('nan')

    def _marker(self, marker_id, marker_type, scale, color, position=None):
        marker = Marker()
        marker.header.frame_id, marker.header.stamp = self.base_frame, self.get_clock().now().to_msg()
        marker.ns, marker.id, marker.type, marker.action = 'ur5e_camera_workspace', marker_id, marker_type, Marker.ADD
        marker.scale.x, marker.scale.y, marker.scale.z = scale
        marker.color.r, marker.color.g, marker.color.b, marker.color.a = color
        if position:
            marker.pose.position.x, marker.pose.position.y, marker.pose.position.z = position
        marker.pose.orientation.w = 1.0
        return marker

    def _publish_markers(self):
        markers = MarkerArray()
        markers.markers.append(self._marker(0, Marker.CUBE, (self.width, self.depth, 0.015), (0.25, 0.45, 0.8, 0.45), (self.center[0], self.center[1], self.table_height - 0.0075)))
        markers.markers.append(self._marker(1, Marker.SPHERE, (0.04, 0.04, 0.04), (1.0, 0.75, 0.0, 1.0), self.center))
        for index, corner in enumerate(self._corners(), start=2):
            markers.markers.append(self._marker(index, Marker.SPHERE, (0.025, 0.025, 0.025), (0.1, 1.0, 0.2, 1.0), corner))
        camera = self._marker(6, Marker.CUBE, (0.10, 0.05, 0.04), (0.35, 0.35, 0.35, 1.0), self.camera_position)
        camera.pose.orientation.x, camera.pose.orientation.y, camera.pose.orientation.z, camera.pose.orientation.w = self.quaternion
        markers.markers.append(camera)
        axis = self._marker(7, Marker.ARROW, (0.008, 0.018, 0.018), (1.0, 0.1, 0.1, 1.0))
        axis.points = [Point(x=point[0], y=point[1], z=point[2]) for point in (self.camera_position, self.center)]
        markers.markers.append(axis)
        frustum = self._marker(8, Marker.LINE_LIST, (0.006, 0.0, 0.0), (1.0, 0.5, 0.0, 0.9))
        tan_h, tan_v = math.tan(self.hfov / 2.0), math.tan(self.vfov / 2.0)
        rings = []
        for distance in (self.near, self.far):
            ring = []
            origin = tuple(self.camera_position[index] + self.forward[index] * distance for index in range(3))
            for sx, sy in ((-1, -1), (-1, 1), (1, 1), (1, -1)):
                ring.append(tuple(origin[index] + self.right[index] * sx * distance * tan_h + self.down[index] * sy * distance * tan_v for index in range(3)))
            rings.append(ring)
        for ring in rings:
            for index in range(4):
                frustum.points.extend([Point(x=ring[index][0], y=ring[index][1], z=ring[index][2]), Point(x=ring[(index + 1) % 4][0], y=ring[(index + 1) % 4][1], z=ring[(index + 1) % 4][2])])
        for index in range(4):
            frustum.points.extend([Point(x=rings[0][index][0], y=rings[0][index][1], z=rings[0][index][2]), Point(x=rings[1][index][0], y=rings[1][index][1], z=rings[1][index][2])])
        markers.markers.append(frustum)
        self.marker_pub.publish(markers)


def main(args=None):
    rclpy.init(args=args)
    node = WorkspaceVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()