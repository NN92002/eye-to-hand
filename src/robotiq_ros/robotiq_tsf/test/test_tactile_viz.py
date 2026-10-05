# Copyright (c) 2026 Robotiq
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#    * Redistributions of source code must retain the above copyright
#      notice, this list of conditions and the following disclaimer.
#
#    * Redistributions in binary form must reproduce the above copyright
#      notice, this list of conditions and the following disclaimer in the
#      documentation and/or other materials provided with the distribution.
#
#    * Neither the name of the copyright holder nor the names of its
#      contributors may be used to endorse or promote products derived from
#      this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Unit tests for scripts/tactile_viz_node (baseline zeroing, noise floor,
marker/heatmap generation, colormap, rpy->quat).

The script has no .py extension, so it is loaded via importlib; the node's
callbacks are driven directly with constructed StaticData messages and the
publishers are replaced with recording stubs — no executor or driver needed.
"""
import importlib.machinery
import importlib.util
import math
import pathlib

import pytest
import rclpy
from rclpy.parameter import Parameter
from robotiq_tsf.msg import StaticData

_SCRIPT = (
    pathlib.Path(__file__).resolve().parent.parent / "scripts" / "tactile_viz_node"
)
_loader = importlib.machinery.SourceFileLoader("tactile_viz_node", str(_SCRIPT))
_spec = importlib.util.spec_from_loader("tactile_viz_node", _loader)
viz = importlib.util.module_from_spec(_spec)
_loader.exec_module(viz)

ROWS, COLS, TAXELS = viz._GRID_ROWS, viz._GRID_COLS, viz._TAXELS
NAVY = (0.05, 0.05, 0.35)
RED = (0.9, 0.1, 0.1)


class _RecordingPub:
    def __init__(self):
        self.msgs = []

    def publish(self, msg):
        self.msgs.append(msg)


@pytest.fixture(scope="module", autouse=True)
def ros():
    rclpy.init()
    yield
    rclpy.try_shutdown()


def make_node(**params):
    overrides = [Parameter(k, value=v) for k, v in params.items()]
    node = viz.TactileVizNode(
        parameter_overrides=overrides, start_parameter_services=False
    )
    node._marker_pub = _RecordingPub()
    node._heatmap_pubs = [_RecordingPub(), _RecordingPub()]
    return node


def make_msg(f0=0, f1=None):
    """StaticData with constant (or per-taxel list) values per finger."""
    msg = StaticData()
    for fi, v in enumerate((f0, f0 if f1 is None else f1)):
        values = [v] * TAXELS if isinstance(v, int) else list(v)
        msg.taxels[fi].values = values
    return msg


def marker_values(node):
    """Colors of the last published MarkerArray, as [(r, g, b)] per finger."""
    array = node._marker_pub.msgs[-1]
    assert len(array.markers) == 2 * TAXELS
    out = [[], []]
    for m in array.markers:
        fi = int(m.ns[-1])
        out[fi].append((m.color.r, m.color.g, m.color.b))
    return out


def test_colormap_endpoints_and_clamp():
    assert viz._colormap(0.0) == pytest.approx(NAVY)
    assert viz._colormap(1.0) == pytest.approx(RED)
    assert viz._colormap(-5.0) == pytest.approx(NAVY)
    assert viz._colormap(5.0) == pytest.approx(RED)
    assert viz._colormap(1.0 / 3.0) == pytest.approx((0.0, 0.7, 0.9))
    assert viz._colormap(2.0 / 3.0) == pytest.approx((1.0, 0.95, 0.2))


def test_colormap_is_continuous():
    prev = viz._colormap(0.0)
    for i in range(1, 101):
        cur = viz._colormap(i / 100.0)
        assert all(abs(a - b) < 0.05 for a, b in zip(prev, cur))
        prev = cur


def test_rpy_to_quat():
    q = viz._rpy_to_quat(0.0, 0.0, 0.0)
    assert (q.w, q.x, q.y, q.z) == pytest.approx((1.0, 0.0, 0.0, 0.0))
    q = viz._rpy_to_quat(0.0, 0.0, math.pi)
    assert (q.w, q.x, q.y, q.z) == pytest.approx((0.0, 0.0, 0.0, 1.0), abs=1e-9)
    q = viz._rpy_to_quat(math.pi / 2, 0.0, 0.0)
    assert (q.w, q.x, q.y, q.z) == pytest.approx(
        (math.cos(math.pi / 4), math.sin(math.pi / 4), 0.0, 0.0)
    )
    # General-case golden (ZYX composition), pinned so a future swap to a
    # library helper cannot silently change convention.
    q = viz._rpy_to_quat(0.3, -0.7, 1.1)
    assert (q.w, q.x, q.y, q.z) == pytest.approx(
        (0.765062179348, 0.296891540058, -0.215672410090, 0.529169808944)
    )
    assert q.w**2 + q.x**2 + q.y**2 + q.z**2 == pytest.approx(1.0)


def test_markers_zero_during_baseline_capture():
    node = make_node(baseline_frames=3)
    try:
        node._on_data(make_msg(600))
        for finger in marker_values(node):
            assert all(c == pytest.approx(NAVY) for c in finger)
    finally:
        node.destroy_node()


def test_baseline_subtraction_and_noise_floor():
    node = make_node(baseline_frames=2, noise_floor=50.0, vmin=0.0, vmax=700.0)
    try:
        node._on_data(make_msg(1000))
        node._on_data(make_msg(1000))  # baseline = 1000 captured

        node._on_data(make_msg(1049))  # residual 49 < floor -> zero
        for finger in marker_values(node):
            assert all(c == pytest.approx(NAVY) for c in finger)

        node._on_data(make_msg(1050))  # residual 50 == floor -> kept
        expected = viz._colormap(50.0 / 700.0)
        for finger in marker_values(node):
            assert all(c == pytest.approx(expected) for c in finger)

        node._on_data(make_msg(500))  # below baseline -> clamped to 0
        for finger in marker_values(node):
            assert all(c == pytest.approx(NAVY) for c in finger)
    finally:
        node.destroy_node()


def test_rezero_recaptures_baseline():
    node = make_node(baseline_frames=1, noise_floor=0.0)
    try:
        node._on_data(make_msg(100))  # baseline = 100
        node._on_data(make_msg(300))  # residual 200
        assert marker_values(node)[0][0] != pytest.approx(NAVY)

        node._on_zero(None)
        node._on_data(make_msg(300))  # new baseline = 300, shows zeros
        node._on_data(make_msg(300))  # residual 0
        for finger in marker_values(node):
            assert all(c == pytest.approx(NAVY) for c in finger)
    finally:
        node.destroy_node()


def test_no_zero_on_start_uses_raw_values():
    node = make_node(zero_on_start=False, noise_floor=0.0, vmin=0.0, vmax=700.0)
    try:
        node._on_data(make_msg(700))
        for finger in marker_values(node):
            assert all(c == pytest.approx(RED) for c in finger)
    finally:
        node.destroy_node()


def test_marker_geometry_and_ids():
    node = make_node(zero_on_start=False)
    try:
        node._on_data(make_msg(0))
        array = node._marker_pub.msgs[-1]
        f0 = [m for m in array.markers if m.ns == "tactile_finger_0"]
        assert sorted(m.id for m in f0) == list(range(TAXELS))
        assert all(m.header.frame_id == "tactile_finger_0" for m in f0)
        pitch_y, pitch_z = 0.022 / 4, 0.037 / 7
        y_half = (COLS - 1) * 0.5 * pitch_y
        z_half = (ROWS - 1) * 0.5 * pitch_z
        top_left = next(m for m in f0 if m.id == 0)  # r=0, c=0
        assert top_left.pose.position.y == pytest.approx(y_half)
        assert top_left.pose.position.z == pytest.approx(z_half)
        bottom_right = next(m for m in f0 if m.id == TAXELS - 1)
        assert bottom_right.pose.position.y == pytest.approx(-y_half)
        assert bottom_right.pose.position.z == pytest.approx(-z_half)
    finally:
        node.destroy_node()


def test_flip_finger_1_reverses_columns():
    ramp = list(range(TAXELS))  # distinct value per taxel
    node = make_node(zero_on_start=False, flip_finger_1=True, noise_floor=0.0)
    try:
        node._on_data(make_msg(ramp, ramp))
        array = node._marker_pub.msgs[-1]
        by_id = {m.id: m for m in array.markers if m.ns == "tactile_finger_1"}
        ref = {m.id: m for m in array.markers if m.ns == "tactile_finger_0"}
        for r in range(ROWS):
            for c in range(COLS):
                flipped = by_id[r * COLS + c].color
                straight = ref[r * COLS + (COLS - 1 - c)].color
                assert (flipped.r, flipped.g, flipped.b) == pytest.approx(
                    (straight.r, straight.g, straight.b)
                )
    finally:
        node.destroy_node()


def test_heatmap_image_shape_and_scaling():
    node = make_node(
        zero_on_start=False,
        heatmap_scale_px=10,
        heatmap_publish_hz=0.0,
        heatmap_floor=0.0,
    )
    try:
        node._on_data(make_msg(100))
        for pub in node._heatmap_pubs:
            img = pub.msgs[-1]
            assert (img.height, img.width) == (ROWS * 10, COLS * 10)
            assert img.encoding == "rgb8"
            assert img.step == img.width * 3
            assert len(img.data) == img.height * img.step
    finally:
        node.destroy_node()


def test_heatmap_autoscale_tracks_running_max():
    node = make_node(
        zero_on_start=False,
        heatmap_scale_px=1,
        heatmap_publish_hz=0.0,
        heatmap_floor=0.0,
        heatmap_autoscale=True,
    )

    def px(t):
        return bytes(round(v * 255) for v in viz._colormap(t))

    try:
        node._on_data(make_msg(200))  # max: 200 -> full scale
        assert bytes(node._heatmap_pubs[0].msgs[-1].data[0:3]) == px(1.0)

        node._on_data(make_msg(400))  # new max: 400, still full scale
        assert bytes(node._heatmap_pubs[0].msgs[-1].data[0:3]) == px(1.0)

        node._on_data(make_msg(200))  # now mid-scale vs max 400
        assert bytes(node._heatmap_pubs[0].msgs[-1].data[0:3]) == px(0.5)
    finally:
        node.destroy_node()


def test_heatmap_rate_limit():
    node = make_node(zero_on_start=False, heatmap_publish_hz=1.0, heatmap_scale_px=1)
    try:
        node._on_data(make_msg(100))
        node._on_data(make_msg(100))  # within 1s window -> throttled
        assert len(node._heatmap_pubs[0].msgs) == 1
        assert len(node._marker_pub.msgs) == 2  # markers are never throttled
    finally:
        node.destroy_node()


def test_heatmaps_disabled():
    node = make_node(zero_on_start=False, publish_2d_heatmaps=False)
    try:
        node._heatmap_pubs = []
        node._on_data(make_msg(100))
        assert len(node._marker_pub.msgs) == 1
    finally:
        node.destroy_node()


def test_wrong_taxel_count_is_dropped():
    # rosidl's fixed-size array can't carry a short payload, so exercise the
    # guard with a duck-typed message (e.g. a future variable-length source).
    class FakeTaxels:
        def __init__(self, n):
            self.values = [0] * n

    class FakeMsg:
        def __init__(self):
            self.taxels = [FakeTaxels(TAXELS), FakeTaxels(TAXELS - 1)]

    node = make_node(zero_on_start=False)
    try:
        node._on_data(FakeMsg())
        assert node._marker_pub.msgs == []  # dropped, nothing published
    finally:
        node.destroy_node()


def test_publish_static_tfs():
    class RecordingBroadcaster:
        def __init__(self):
            self.transforms = []

        def sendTransform(self, transforms):
            self.transforms.extend(transforms)

    node = make_node(zero_on_start=False)
    try:
        bc = RecordingBroadcaster()
        node._tf_broadcaster = bc
        node._publish_static_tfs()
        assert [t.child_frame_id for t in bc.transforms] == [
            "tactile_finger_0",
            "tactile_finger_1",
            "tactile_tip",
        ]
        assert all(t.header.frame_id == "tactile_world" for t in bc.transforms)
        f0, f1, tip = bc.transforms
        assert f0.transform.translation.x == pytest.approx(-0.0425)
        assert f1.transform.translation.x == pytest.approx(0.0425)
        assert tip.transform.translation.z == pytest.approx(0.0209)
        assert f0.transform.rotation.w == pytest.approx(1.0)
        q = f1.transform.rotation  # finger 1 yawed pi to face finger 0
        assert (q.w, q.x, q.y, abs(q.z)) == pytest.approx(
            (0.0, 0.0, 0.0, 1.0), abs=1e-9
        )
    finally:
        node.destroy_node()


def test_heatmap_pixel_orientation():
    # Ramp value = r*COLS + c: every pixel unique, pinning the image's
    # row/column orientation, not just pixel (0,0).
    ramp = list(range(TAXELS))
    node = make_node(
        zero_on_start=False,
        heatmap_scale_px=1,
        heatmap_publish_hz=0.0,
        heatmap_autoscale=False,
        vmin=0.0,
        vmax=float(TAXELS - 1),
        noise_floor=0.0,
    )
    try:
        node._on_data(make_msg(ramp, ramp))
        img = node._heatmap_pubs[0].msgs[-1]

        def px(r, c):
            o = (r * img.width + c) * 3
            return bytes(img.data[o : o + 3])

        def expected(r, c):
            t = (r * COLS + c) / (TAXELS - 1)
            return bytes(round(v * 255) for v in viz._colormap(t))

        for r, c in [(0, 0), (0, 3), (2, 1), (4, 3), (6, 0), (6, 3)]:
            assert px(r, c) == expected(r, c), f"pixel ({r},{c})"
    finally:
        node.destroy_node()
