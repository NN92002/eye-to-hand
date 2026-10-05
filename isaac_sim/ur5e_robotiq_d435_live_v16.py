"""
Isaac Sim 6.1
UR5e + Robotiq 2F-140 + Workspace + D435 + LIVE Joint/Gripper GUI (v18 - gripper controls visible at top)

Main Viewport:
    Perspective view of entire scene

Second Viewport:
    D435 RGB camera view

Run:
    cd /home/neaf6000pro/isaacsim

    ./python.sh \
    /home/n/eye-to-hand/isaac_sim/ur5e_camera_scene.py
"""

from pathlib import Path
import re

import numpy as np
import threading
import queue

# Optional OpenCV window for live depth visualization.
try:
    import cv2
except ImportError:
    cv2 = None

# ============================================================
# Start Isaac Sim
# ============================================================

from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
    }
)


# ============================================================
# Isaac imports
# ============================================================

import carb
import omni.usd
import omni.ui as ui
from isaacsim.gui.components import ui_utils

from pxr import UsdGeom, UsdPhysics

from isaacsim.core.api import World
from isaacsim.core.api.objects import FixedCuboid
from isaacsim.core.prims import SingleArticulation
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.sensors.camera import Camera
from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

from isaacsim.asset.importer.urdf import (
    URDFImporter,
    URDFImporterConfig,
)

from isaacsim.core.rendering_manager import ViewportManager


# ============================================================
# Paths
# ============================================================

WORKSPACE_ROOT = Path(
    "/home/n/eye-to-hand"
)

SOURCE_URDF = (
    WORKSPACE_ROOT
    / "isaac_sim"
    / "ur5e_robotiq_2f140.urdf"
)

RESOLVED_URDF = Path(
    "/tmp/ur5e_robotiq_2f140_isaac.urdf"
)

UR_DESCRIPTION_DIR = (
    WORKSPACE_ROOT
    / "src"
    / "universal_robot"
)

ROBOTIQ_DESCRIPTION_DIR = (
    WORKSPACE_ROOT
    / "src"
    / "robotiq_ros"
    / "grippers"
    / "robotiq_description"
)

UR5E_USD = Path(
    "/tmp/ur5e_robotiq_2f140_final.usd"
)


# ============================================================
# Workspace geometry
# ============================================================

# Workspace dimensions
TABLE_WIDTH = 0.70
TABLE_DEPTH = 0.70
TABLE_THICKNESS = 0.03

# IMPORTANT:
# UR5e base = world origin
#
# For now, make the work surface approximately level with
# the robot base.
TABLE_HEIGHT = 0.0

# Ground plane: its TOP surface is exactly world Z = 0.0 m.
# This gives downward-looking camera rays a finite surface to hit.
GROUND_SIZE_X = 4.0
GROUND_SIZE_Y = 4.0
GROUND_THICKNESS = 0.02
GROUND_TOP_Z = 0.0

# Depth display range in meters. Values outside this range are clipped.
DEPTH_MIN_M = 0.20
DEPTH_MAX_M = 3.00
DEPTH_WINDOW_NAME = "D435 Depth (meters)"

# Workspace center in UR5e base/world coordinates.
WORKSPACE_CENTER = np.array(
    [
        0.45,
        0.0,
        TABLE_HEIGHT,
    ],
    dtype=np.float32,
)


# ============================================================
# D435 pose
# ============================================================

# ------------------------------------------------------------
# Camera position in UR5e/world coordinates
#
# Current temporary setup:
#
#        +Z
#         ^
#         |
#      D435
#        \
#         \
#          v
#     workspace
#
# D435 is placed on the -Y side of the workspace.
# ------------------------------------------------------------

CAMERA_POSITION = np.array(
    [
        0.80,    # X
        0.00,   # Y
        1.5,    # Z
    ],
    dtype=np.float32,
)

# Camera looks at workspace center.
CAMERA_TARGET = np.array(
    [
        0.45,
        0.00,
        0.00,
    ],
    dtype=np.float32,
)


# ============================================================
# D435 camera settings
# ============================================================

CAMERA_RESOLUTION = (
    640,
    480,
)

CAMERA_FREQUENCY = 30

D435_CAMERA_PATH = (
    "/D435/camera_color_optical_frame"
)


# ============================================================
# D435 body visualization
# ============================================================

# Approximate body size only for visualization.
#
# This is NOT intended to be an exact CAD model.

D435_BODY_SIZE = np.array(
    [
        0.090,
        0.025,
        0.025,
    ],
    dtype=np.float32,
)


# ============================================================
# UR5e joint control
# ============================================================

UR5E_JOINT_NAMES = [
    "shoulder_pan_joint",
    "shoulder_lift_joint",
    "elbow_joint",
    "wrist_1_joint",
    "wrist_2_joint",
    "wrist_3_joint",
]

# Convenient initial/home pose for testing (degrees).
UR5E_HOME_DEG = np.array(
    [0.0, -90.0, 90.0, -90.0, -90.0, 0.0],
    dtype=np.float32,
)

# GUI range. The imported URDF/robot controller remains the authority for
# the actual robot limits. This wide range is useful for simulation testing.
GUI_JOINT_MIN_DEG = -180.0
GUI_JOINT_MAX_DEG = 180.0

# Thread-safe bridge:
# Tkinter runs in its own thread; ONLY the Isaac Sim main thread touches
# the articulation. This is also the seam where U-Arm commands can enter later.
JOINT_COMMAND_QUEUE = queue.Queue()
JOINT_STATE_QUEUE = queue.Queue(maxsize=1)

ROBOTIQ_MAIN_JOINT = "robotiq_finger_joint"
# Isaac Sim 6.1 safety strategy: the URDF <mimic> tags are removed only from
# the temporary Isaac-resolved URDF.  We then command every movable 2F-140
# linkage joint explicitly with the same ratios as the original URDF.
ROBOTIQ_JOINT_MULTIPLIERS = {
    "robotiq_finger_joint": 1.0,
    "robotiq_left_inner_knuckle_joint": -1.0,
    "robotiq_left_inner_finger_joint": 1.0,
    "robotiq_right_outer_knuckle_joint": -1.0,
    "robotiq_right_inner_knuckle_joint": -1.0,
    "robotiq_right_inner_finger_joint": 1.0,
}
GRIPPER_COMMAND_QUEUE = queue.Queue()
GRIPPER_STATE_QUEUE = queue.Queue(maxsize=1)
GRIPPER_MIN_RAD = 0.0
GRIPPER_MAX_RAD = 0.70


def push_latest_state(state_deg):
    """Publish the latest joint state to the GUI without blocking simulation."""
    try:
        while True:
            JOINT_STATE_QUEUE.get_nowait()
    except queue.Empty:
        pass
    try:
        JOINT_STATE_QUEUE.put_nowait(np.asarray(state_deg, dtype=np.float32))
    except queue.Full:
        pass


def enqueue_joint_command(target_deg, source="gui"):
    """Common joint-command input. U-Arm can call this later too."""
    target_deg = np.asarray(target_deg, dtype=np.float32)
    if target_deg.shape != (6,):
        raise ValueError("UR5e target must contain exactly 6 joint angles.")
    JOINT_COMMAND_QUEUE.put((source, target_deg.copy()))


def push_latest_gripper_state(position_rad):
    try:
        while True:
            GRIPPER_STATE_QUEUE.get_nowait()
    except queue.Empty:
        pass
    try:
        GRIPPER_STATE_QUEUE.put_nowait(float(position_rad))
    except queue.Full:
        pass


def enqueue_gripper_command(close_percent, source="gui"):
    """0 percent = open, 100 percent = closed."""
    value = float(np.clip(close_percent, 0.0, 100.0))
    GRIPPER_COMMAND_QUEUE.put((source, value))


def start_joint_control_gui():
    """Create an Isaac Sim native six-axis UR5e LIVE control window.

    Uses Isaac Sim's ui_utils float-field/slider builder instead of constructing
    omni.ui.FloatSlider directly.  This avoids pybind conversion issues with
    NumPy scalar values in Isaac Sim 6.1 / Python 3.12.
    """
    carb.log_info("Creating native Isaac Sim UR5e Joint Control window...")

    window = ui.Window("UR5e + Robotiq 2F-140 LIVE Control", width=760, height=650)
    models = []
    current_labels = []
    state = {"latest": None, "gripper_latest": None, "suppress_live_send": False, "suppress_gripper_send": False}
    gripper_model_holder = {"model": None}
    gripper_current_label_holder = {"label": None}

    def get_targets():
        values = []
        for model in models:
            # AbstractValueModel API returns a native Python float.
            values.append(float(model.get_value_as_float()))
        return np.asarray(values, dtype=np.float32)

    def send_live_target(_model=None):
        """Send the newest six-axis target whenever a slider/field changes."""
        if state["suppress_live_send"] or len(models) != 6:
            return
        try:
            target = get_targets()
            enqueue_joint_command(target, source="omni_ui_live")
            status_label.text = "LIVE: " + np.array2string(target, precision=1) + " deg"
        except Exception as error:
            status_label.text = f"Error: {error}"
            carb.log_error(f"UR5e GUI live control failed: {error!r}")

    def home():
        try:
            state["suppress_live_send"] = True
            for model, value in zip(models, UR5E_HOME_DEG.tolist()):
                model.set_value(float(value))
            state["suppress_live_send"] = False
            enqueue_joint_command(np.asarray(UR5E_HOME_DEG, dtype=np.float32), source="omni_ui_home")
            status_label.text = "Home target sent"
        except Exception as error:
            state["suppress_live_send"] = False
            status_label.text = f"Error: {error}"
            carb.log_error(f"UR5e GUI Home failed: {error!r}")

    def read_current():
        latest = state["latest"]
        if latest is None:
            status_label.text = "No joint state received yet"
            return
        try:
            state["suppress_live_send"] = True
            for model, value in zip(models, latest.tolist()):
                model.set_value(float(value))
            state["suppress_live_send"] = False
            status_label.text = "Targets synchronized to current robot pose"
        except Exception as error:
            state["suppress_live_send"] = False
            status_label.text = f"Error: {error}"
            carb.log_error(f"UR5e GUI Read Current failed: {error!r}")

    def send_live_gripper(_model=None):
        if state["suppress_gripper_send"]:
            return
        model = gripper_model_holder["model"]
        if model is None:
            return
        try:
            close_percent = float(model.get_value_as_float())
            enqueue_gripper_command(close_percent, source="omni_ui_gripper_live")
            status_label.text = f"LIVE gripper: {close_percent:.1f}% closed"
        except Exception as error:
            carb.log_error(f"Robotiq GUI live control failed: {error!r}")

    def gripper_open():
        model = gripper_model_holder["model"]
        if model is not None:
            model.set_value(0.0)

    def gripper_close():
        model = gripper_model_holder["model"]
        if model is not None:
            model.set_value(100.0)

    with window.frame:
        with ui.VStack(spacing=6, height=0):
            # Keep the gripper controls at the TOP so they remain visible even
            # when this window is docked into a short bottom panel.
            ui.Label("Robotiq 2F-140 Gripper - LIVE", height=26, alignment=ui.Alignment.CENTER)
            with ui.HStack(height=38, spacing=8):
                ui.Label("Open 0%  <->  Close 100%", width=175)
                gripper_field_model, _gripper_slider_model = ui_utils.combo_floatfield_slider_builder(
                    label="",
                    default_val=0.0,
                    min=0.0,
                    max=100.0,
                    step=1.0,
                    tooltip=["2F-140 close percentage", "2F-140 close percentage"],
                )
                gripper_model_holder["model"] = gripper_field_model
                gripper_current_label = ui.Label("current: --.- %", width=145)
                gripper_current_label_holder["label"] = gripper_current_label
            gripper_field_model.add_value_changed_fn(send_live_gripper)
            with ui.HStack(height=32, spacing=10):
                ui.Spacer()
                ui.Button("OPEN", width=110, clicked_fn=gripper_open)
                ui.Button("CLOSE", width=110, clicked_fn=gripper_close)
                ui.Spacer()

            ui.Separator(height=4)
            ui.Label("UR5e Joint Angle Control - LIVE", height=24, alignment=ui.Alignment.CENTER)
            ui.Label(
                "Command: degree | Isaac Sim: radian",
                height=18,
                alignment=ui.Alignment.CENTER,
            )

            home_values = [float(v) for v in UR5E_HOME_DEG.tolist()]
            for joint_name, home_deg in zip(UR5E_JOINT_NAMES, home_values):
                with ui.HStack(height=34, spacing=8):
                    ui.Label(str(joint_name), width=175)
                    field_model, _slider_model = ui_utils.combo_floatfield_slider_builder(
                        label="",
                        default_val=float(home_deg),
                        min=float(GUI_JOINT_MIN_DEG),
                        max=float(GUI_JOINT_MAX_DEG),
                        step=0.1,
                        tooltip=[f"{joint_name} target (deg)", f"{joint_name} target (deg)"],
                    )
                    models.append(field_model)
                    label = ui.Label("current: --.- deg", width=145)
                    current_labels.append(label)

            for model in models:
                model.add_value_changed_fn(send_live_target)

            with ui.HStack(height=32, spacing=10):
                ui.Spacer()
                ui.Button("Home", width=110, clicked_fn=home)
                ui.Button("Read Current", width=130, clicked_fn=read_current)
                ui.Spacer()

            status_label = ui.Label("LIVE control ready - arm + gripper", height=22, alignment=ui.Alignment.CENTER)

    window.visible = True

    def refresh():
        latest = None
        try:
            while True:
                latest = JOINT_STATE_QUEUE.get_nowait()
        except queue.Empty:
            pass
        if latest is None:
            return

        latest = np.asarray(latest, dtype=np.float32)
        state["latest"] = latest.copy()
        for label, value in zip(current_labels, latest.tolist()):
            label.text = f"current: {float(value):7.2f} deg"
        gripper_latest = None
        try:
            while True:
                gripper_latest = GRIPPER_STATE_QUEUE.get_nowait()
        except queue.Empty:
            pass
        if gripper_latest is not None:
            state["gripper_latest"] = float(gripper_latest)
            pct = 100.0 * (float(gripper_latest) - GRIPPER_MIN_RAD) / (GRIPPER_MAX_RAD - GRIPPER_MIN_RAD)
            pct = float(np.clip(pct, 0.0, 100.0))
            label = gripper_current_label_holder["label"]
            if label is not None:
                label.text = f"current: {pct:6.1f} %"

    carb.log_info("Native Isaac Sim UR5e Joint Control window created.")
    return {"window": window, "refresh": refresh, "models": models}

def create_ur5e_articulation(articulation_root, world):
    """Register UR5e with World BEFORE world.reset().

    Isaac Sim builds the PhysX tensor articulation view during reset.  Creating
    and manually initializing SingleArticulation after reset can leave the
    tensor backend with no matching articulation.
    """
    if not articulation_root:
        raise RuntimeError("UR5e ArticulationRootAPI prim was not found.")

    carb.log_info(f"Registering UR5e articulation before reset: {articulation_root}")
    robot = SingleArticulation(
        prim_path=articulation_root,
        name="ur5e_joint_controller",
    )
    world.scene.add(robot)
    return robot


def _require_finite(values, label):
    arr = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        bad = np.where(~np.isfinite(arr))[0].tolist()
        raise RuntimeError(f"{label} contains NaN/Inf at indices {bad}: {arr}")
    return arr


def finalize_ur5e_articulation(robot):
    """Validate UR5e + Robotiq after World.reset().

    Mimic is deliberately disabled in the temporary Isaac URDF.  Therefore all
    six movable Robotiq linkage joints must appear as ordinary articulation DOFs.
    """
    dof_names = list(robot.dof_names)
    carb.log_info(f"UR5e + Robotiq DOFs ({len(dof_names)}): {dof_names}")
    if not dof_names:
        raise RuntimeError("Articulation initialized but contains no DOFs.")

    arm_indices = []
    for name in UR5E_JOINT_NAMES:
        if name not in dof_names:
            raise RuntimeError(f"Required UR5e joint '{name}' not found. Imported DOFs: {dof_names}")
        arm_indices.append(dof_names.index(name))
    arm_indices = np.asarray(arm_indices, dtype=np.int32)

    gripper_names = []
    gripper_indices = []
    gripper_multipliers = []
    for name, multiplier in ROBOTIQ_JOINT_MULTIPLIERS.items():
        if name in dof_names:
            gripper_names.append(name)
            gripper_indices.append(dof_names.index(name))
            gripper_multipliers.append(multiplier)
        else:
            carb.log_warn(f"Robotiq DOF not found: {name}")

    if ROBOTIQ_MAIN_JOINT not in gripper_names:
        raise RuntimeError(f"Robotiq main joint '{ROBOTIQ_MAIN_JOINT}' was not imported. DOFs: {dof_names}")

    gripper_indices = np.asarray(gripper_indices, dtype=np.int32)
    gripper_multipliers = np.asarray(gripper_multipliers, dtype=np.float32)
    carb.log_info(f"Robotiq controlled DOFs: {list(zip(gripper_names, gripper_indices.tolist(), gripper_multipliers.tolist()))}")

    current = robot.get_joint_positions()
    if current is None:
        raise RuntimeError("get_joint_positions() returned None after reset.")
    current = _require_finite(current, "Initial articulation joint positions")
    push_latest_state(np.rad2deg(current[arm_indices]))
    main_slot = gripper_names.index(ROBOTIQ_MAIN_JOINT)
    push_latest_gripper_state(current[gripper_indices[main_slot]])
    carb.log_info("Initial UR5e + Robotiq articulation state is finite.")
    return robot, arm_indices, gripper_indices, gripper_multipliers, gripper_names


def process_joint_commands(robot, joint_indices):
    """
    Apply pending six-axis commands from the GUI (and later U-Arm).

    The GUI/U-Arm layer uses degrees. Conversion to radians happens only here.
    """
    newest = None
    while True:
        try:
            newest = JOINT_COMMAND_QUEUE.get_nowait()
        except queue.Empty:
            break

    if newest is None:
        return

    source, target_deg = newest
    target_rad = np.deg2rad(target_deg).astype(np.float32)

    # Isaac Sim 6.1 SingleArticulation does not expose
    # set_joint_position_targets() directly. Send the command through its
    # ArticulationController instead, so PhysX drives the joints rather than
    # teleporting them.
    controller = robot.get_articulation_controller()
    if controller is None:
        raise RuntimeError("UR5e articulation controller is unavailable.")

    action = ArticulationAction(
        joint_positions=target_rad,
        joint_indices=joint_indices,
    )
    controller.apply_action(action)
    carb.log_info(
        f"UR5e target from {source}: "
        + np.array2string(target_deg, precision=1)
        + " deg"
    )


def process_gripper_commands(robot, gripper_indices, gripper_multipliers):
    if gripper_indices is None or len(gripper_indices) == 0:
        return
    newest = None
    while True:
        try:
            newest = GRIPPER_COMMAND_QUEUE.get_nowait()
        except queue.Empty:
            break
    if newest is None:
        return

    source, close_percent = newest
    main_target = GRIPPER_MIN_RAD + (float(close_percent) / 100.0) * (GRIPPER_MAX_RAD - GRIPPER_MIN_RAD)
    targets = (float(main_target) * np.asarray(gripper_multipliers, dtype=np.float32)).astype(np.float32)
    _require_finite(targets, "Robotiq command")

    controller = robot.get_articulation_controller()
    if controller is None:
        raise RuntimeError("UR5e/Robotiq articulation controller is unavailable.")
    controller.apply_action(ArticulationAction(joint_positions=targets, joint_indices=gripper_indices))
    carb.log_info(f"Robotiq target from {source}: {close_percent:.1f}% -> main {main_target:.4f} rad")


def publish_joint_state(robot, joint_indices, gripper_indices=None, gripper_names=None):
    current = robot.get_joint_positions()
    if current is None:
        return
    current = np.asarray(current, dtype=np.float64)
    if not np.all(np.isfinite(current)):
        bad = np.where(~np.isfinite(current))[0].tolist()
        carb.log_error(f"ARTICULATION BECAME NON-FINITE at DOF indices {bad}; skipping GUI state update.")
        return
    push_latest_state(np.rad2deg(current[joint_indices]))
    if gripper_indices is not None and len(gripper_indices) and gripper_names and ROBOTIQ_MAIN_JOINT in gripper_names:
        main_slot = gripper_names.index(ROBOTIQ_MAIN_JOINT)
        push_latest_gripper_state(current[gripper_indices[main_slot]])


# ============================================================
# Utility
# ============================================================

def wait_updates(count=10):

    for _ in range(count):
        simulation_app.update()


# ============================================================
# Prepare URDF
# ============================================================

def prepare_urdf():

    carb.log_info(
        "========================================"
    )

    carb.log_info(
        "Preparing UR5e URDF"
    )

    carb.log_info(
        "========================================"
    )

    if not SOURCE_URDF.exists():

        raise FileNotFoundError(
            f"URDF not found: {SOURCE_URDF}"
        )

    if not UR_DESCRIPTION_DIR.exists():

        raise FileNotFoundError(
            f"UR description not found: "
            f"{UR_DESCRIPTION_DIR}"
        )

    text = SOURCE_URDF.read_text()

    old_prefix = (
        "package://ur_description/"
    )

    new_prefix = (
        f"{UR_DESCRIPTION_DIR.as_posix()}/"
    )

    count = text.count(
        old_prefix
    )

    carb.log_info(
        f"Found {count} package URIs."
    )

    text = text.replace(
        old_prefix,
        new_prefix,
    )

    if old_prefix in text:

        raise RuntimeError(
            "package://ur_description "
            "still exists."
        )

    RESOLVED_URDF.write_text(
        text
    )

    carb.log_info(
        f"Resolved URDF: {RESOLVED_URDF}"
    )

    return RESOLVED_URDF


# ============================================================
# Import UR5e
# ============================================================

def import_ur5e():
    """
    Import UR5e with the Isaac Sim 6.1 URDFImporter pipeline.

    Important:
    - Uses the already-expanded URDF generated by ROS xacro.
    - Resolves package://ur_description mesh paths to absolute paths.
    - Keeps the URDF importer native PhysX schemas (RigidBody/Joint/Articulation).
    - Fixed base + position drives are authored during import.
    - References the resulting USD into the current scene under /World/UR5e.
    """
    source_urdf = Path(
        "/home/n/eye-to-hand/isaac_sim/ur5e_robotiq_2f140.urdf"
    )
    if not source_urdf.exists():
        raise FileNotFoundError(
            f"Expanded URDF not found: {source_urdf}\n"
            "Generate it first with ROS2 xacro."
        )

    mesh_root = Path(
        "/home/n/eye-to-hand/src/universal_robot"
    )
    if not mesh_root.exists():
        raise FileNotFoundError(
            f"Universal Robot source directory not found: {mesh_root}"
        )

    # The expanded URDF still contains package://ur_description/...
    # Resolve those URLs so the standalone Isaac Sim process does not need ROS.
    resolved_urdf = Path(
        "/home/n/eye-to-hand/isaac_sim/ur5e_robotiq_2f140_isaac_resolved.urdf"
    )
    urdf_text = source_urdf.read_text()
    urdf_text = urdf_text.replace(
        "package://ur_description/",
        mesh_root.as_posix().rstrip("/") + "/",
    )
    robotiq_mesh_root = Path("/home/n/eye-to-hand/src/robotiq_ros/grippers/robotiq_description")
    if not robotiq_mesh_root.exists():
        raise FileNotFoundError(f"Robotiq description directory not found: {robotiq_mesh_root}")
    urdf_text = urdf_text.replace(
        "package://robotiq_description/",
        robotiq_mesh_root.as_posix().rstrip("/") + "/",
    )

    # IMPORTANT: Isaac Sim 6.1 uses a different mimic pipeline.  For this
    # Robotiq linkage we avoid that path entirely: remove only the self-closing
    # <mimic .../> tags from the temporary Isaac URDF, then explicitly command
    # all linkage joints using ROBOTIQ_JOINT_MULTIPLIERS.  The source URDF is
    # never modified.
    mimic_count = len(re.findall(r"<mimic\s+[^>]*/>", urdf_text))
    urdf_text = re.sub(r"\s*<mimic\s+[^>]*/>", "", urdf_text)
    carb.log_info(f"Removed {mimic_count} mimic tags from temporary Isaac URDF.")
    resolved_urdf.write_text(urdf_text)

    output_dir = Path(
        "/home/n/eye-to-hand/isaac_sim/generated_ur5e_robotiq_2f140_nomimic_v17"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    carb.log_info("========================================")
    carb.log_info("Importing UR5e with Isaac Sim 6.1 URDFImporter")
    carb.log_info(f"URDF: {resolved_urdf}")
    carb.log_info(f"USD output: {output_dir}")
    carb.log_info("========================================")

    config = URDFImporterConfig(
        urdf_path=str(resolved_urdf),
        usd_path=str(output_dir),
        merge_fixed_joints=False,
        merge_mesh=False,
        debug_mode=False,
        collision_from_visuals=False,
        allow_self_collision=False,
        robot_type="Manipulator",
        fix_base=True,
        joint_drive_type="force",
        joint_target_type="position",
        # Conservative initial gains for stable interactive joint control.
        # These can be tuned later for U-Arm teleoperation dynamics.
        override_joint_stiffness=400.0,
        override_joint_damping=40.0,
        # IMPORTANT for this Isaac Sim 6.1 installation:
        # keep the importer-authored PhysX USD. Asset Transformer / multi-physics
        # conversion was producing IsaacRobotAPI/IsaacLinkAPI without PhysX bodies.
        run_asset_transformer=False,
        run_multi_physics_conversion=False,
    )

    importer = URDFImporter(config)
    output_path = importer.import_urdf()

    if not output_path:
        raise RuntimeError("URDFImporter returned an empty USD output path.")

    output_path = str(output_path)
    carb.log_info(f"UR5e USD generated: {output_path}")

    # Keep compatibility with the scene's existing startup flow:
    # main() opens the generated robot USD as the active stage, then adds
    # workspace, ground and D435 into that same stage.
    carb.log_info(f"UR5e USD ready to open: {output_path}")
    return output_path

def open_robot_stage(
    usd_path
):

    carb.log_info(
        "Opening UR5e USD..."
    )

    omni.usd.get_context().open_stage(
        usd_path
    )

    # Give USD / materials time to load.
    wait_updates(60)

    stage = (
        omni.usd
        .get_context()
        .get_stage()
    )

    if stage is None:

        raise RuntimeError(
            "Could not open UR5e USD stage."
        )

    UsdGeom.SetStageUpAxis(
        stage,
        UsdGeom.Tokens.z,
    )

    UsdGeom.SetStageMetersPerUnit(
        stage,
        1.0,
    )

    return stage


# ============================================================
# Find robot root
# ============================================================

def find_robot_root(stage):
    """
    Find the visible/top-level UR5e prim. This is useful for logging only.
    It is NOT assumed to be the PhysX articulation root.
    """
    candidates = [
        "/ur5e",
        "/World/ur5e",
        "/World/UR5e",
    ]

    for path in candidates:
        prim = stage.GetPrimAtPath(path)
        if prim.IsValid():
            carb.log_info(f"UR5e visual/root prim = {path}")
            return path

    for child in stage.GetPseudoRoot().GetChildren():
        path = str(child.GetPath())
        if "ur5" in path.lower():
            carb.log_info(f"UR5e visual/root prim = {path}")
            return path

    carb.log_warn("UR5e top-level prim not automatically found.")
    return None


def dump_ur5e_physics(stage):
    """Log joints and physics APIs so importer hierarchy can be diagnosed."""
    carb.log_info("========== UR5E PHYSICS DIAGNOSTICS ==========")
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        name = prim.GetName()
        if "ur5" not in path.lower() and name not in UR5E_JOINT_NAMES and "base" not in name.lower():
            continue
        try:
            rigid = prim.HasAPI(UsdPhysics.RigidBodyAPI)
            art = prim.HasAPI(UsdPhysics.ArticulationRootAPI)
        except Exception:
            rigid = art = False
        body0 = _relationship_targets(prim, "physics:body0")
        body1 = _relationship_targets(prim, "physics:body1")
        carb.log_info(
            f"PHYS PRIM: {path} | type={prim.GetTypeName()} | rigid={rigid} | "
            f"articulation={art} | body0={body0} | body1={body1}"
        )
    carb.log_info("==============================================")


def _relationship_targets(prim, rel_name):
    try:
        rel = prim.GetRelationship(rel_name)
        if rel and rel.IsValid():
            return [str(path) for path in rel.GetTargets()]
    except Exception:
        pass
    return []


def find_articulation_root(stage, robot_hint=None):
    """Find an importer-authored articulation root without requiring RigidBodyAPI."""
    matches = []
    for prim in stage.Traverse():
        try:
            if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
                matches.append(str(prim.GetPath()))
        except Exception:
            pass

    carb.log_info("========== ARTICULATION ROOT SEARCH ==========")
    for path in matches:
        carb.log_info(f"ARTICULATION ROOT: {path}")
    if not matches:
        carb.log_warn("No existing UsdPhysics.ArticulationRootAPI found.")
    carb.log_info("==============================================")

    # In PhysX an articulation root can legitimately live on an Xform above
    # the rigid links, so do NOT reject it merely because it lacks RigidBodyAPI.
    if robot_hint:
        hint = robot_hint.rstrip("/")
        for path in matches:
            if path == hint or path.startswith(hint + "/"):
                return path
    for path in matches:
        if "ur5" in path.lower():
            return path
    return matches[0] if matches else None


def find_joint_physics_parent(stage):
    """Infer the articulation container from the six UR5e joint prims."""
    joint_paths = []
    for prim in stage.Traverse():
        if prim.GetName() in UR5E_JOINT_NAMES:
            joint_paths.append(str(prim.GetPath()))

    carb.log_info(f"UR5e joint prim paths: {joint_paths}")
    if not joint_paths:
        return None

    # Longest common ancestor of the six joint prims.
    split = [p.strip("/").split("/") for p in joint_paths]
    common = []
    for parts in zip(*split):
        if len(set(parts)) != 1:
            break
        common.append(parts[0])
    if not common:
        return None
    return "/" + "/".join(common)


def ensure_ur5e_articulation_root(stage, robot_hint=None):
    """Use only a real importer-authored PhysX articulation root.

    Do not manufacture ArticulationRootAPI on a visual/IsaacRobotAPI container:
    that does not create rigid bodies or joints and causes the PhysX tensor
    backend to report that the path does not match any articulations.
    """
    dump_ur5e_physics(stage)
    root = find_articulation_root(stage, robot_hint)
    if root is None:
        raise RuntimeError(
            "URDF import produced no UsdPhysics.ArticulationRootAPI. "
            "The native PhysX import failed; see UR5E PHYSICS DIAGNOSTICS."
        )
    carb.log_info(f"Using native importer articulation root: {root}")
    return root


# ============================================================
# Workspace
# ============================================================

def create_ground(world, stage):

    carb.log_info("Creating Z=0 ground plane...")

    ground_path = "/Ground"
    prim = stage.GetPrimAtPath(ground_path)
    if prim.IsValid():
        stage.RemovePrim(ground_path)

    ground = FixedCuboid(
        prim_path=ground_path,
        name="ground_plane",
        position=np.array(
            [0.0, 0.0, GROUND_TOP_Z - GROUND_THICKNESS / 2.0],
            dtype=np.float32,
        ),
        scale=np.array(
            [GROUND_SIZE_X, GROUND_SIZE_Y, GROUND_THICKNESS],
            dtype=np.float32,
        ),
        color=np.array([0.18, 0.18, 0.18], dtype=np.float32),
    )
    world.scene.add(ground)
    carb.log_info("Ground created; top surface Z = 0.0 m.")


def create_workspace(
    world,
    stage,
):

    carb.log_info(
        "Creating workspace..."
    )

    # --------------------------------------------------------
    # Remove old objects if necessary
    # --------------------------------------------------------

    for path in [
        "/Worksurface",
        "/WorkspaceCenter",
    ]:

        prim = stage.GetPrimAtPath(
            path
        )

        if prim.IsValid():

            stage.RemovePrim(
                path
            )

    # --------------------------------------------------------
    # Work surface
    # --------------------------------------------------------

    table = FixedCuboid(

        prim_path="/Worksurface",

        name="worksurface",

        position=np.array(
            [
                WORKSPACE_CENTER[0],
                WORKSPACE_CENTER[1],
                TABLE_HEIGHT
                + TABLE_THICKNESS / 2.0,
            ],
            dtype=np.float32,
        ),

        scale=np.array(
            [
                TABLE_WIDTH,
                TABLE_DEPTH,
                TABLE_THICKNESS,
            ],
            dtype=np.float32,
        ),

        color=np.array(
            [
                0.25,
                0.35,
                0.55,
            ],
            dtype=np.float32,
        ),
    )

    world.scene.add(
        table
    )

    # --------------------------------------------------------
    # Workspace center marker
    # --------------------------------------------------------

    marker = FixedCuboid(

        prim_path="/WorkspaceCenter",

        name="workspace_center",

        position=np.array(
            [
                WORKSPACE_CENTER[0],
                WORKSPACE_CENTER[1],
                TABLE_HEIGHT + 0.025,
            ],
            dtype=np.float32,
        ),

        scale=np.array(
            [
                0.05,
                0.05,
                0.05,
            ],
            dtype=np.float32,
        ),

        color=np.array(
            [
                1.0,
                0.65,
                0.0,
            ],
            dtype=np.float32,
        ),
    )

    world.scene.add(
        marker
    )

    carb.log_info(
        "Workspace created."
    )


# ============================================================
# Create visible D435 body
# ============================================================

def create_d435_body(
    world,
    stage,
):

    carb.log_info(
        "Creating visible D435 body..."
    )

    body_path = (
        "/D435/Body"
    )

    prim = stage.GetPrimAtPath(
        body_path
    )

    if prim.IsValid():

        stage.RemovePrim(
            body_path
        )

    # --------------------------------------------------------
    # This cuboid is only a visual marker representing
    # the physical D435 body.
    # --------------------------------------------------------

    body = FixedCuboid(

        prim_path=body_path,

        name="d435_body",

        position=(
            CAMERA_POSITION.copy()
        ),

        scale=(
            D435_BODY_SIZE.copy()
        ),

        color=np.array(
            [
                0.15,
                0.15,
                0.15,
            ],
            dtype=np.float32,
        ),
    )

    world.scene.add(
        body
    )

    carb.log_info(
        f"D435 body position = "
        f"{CAMERA_POSITION.tolist()}"
    )


# ============================================================
# Create D435 camera
# ============================================================

def create_d435():

    carb.log_info(
        "========================================"
    )

    carb.log_info(
        "Creating D435 camera"
    )

    carb.log_info(
        "========================================"
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We deliberately do NOT calculate our own quaternion here.
    #
    # Create camera first, then let Isaac Sim's camera-view
    # utilities point it toward CAMERA_TARGET.
    # --------------------------------------------------------

    camera = Camera(

        prim_path=(
            D435_CAMERA_PATH
        ),

        name="d435_camera",

        position=(
            CAMERA_POSITION.copy()
        ),

        frequency=(
            CAMERA_FREQUENCY
        ),

        resolution=(
            CAMERA_RESOLUTION
        ),
    )

    # --------------------------------------------------------
    # Approximate D435 intrinsics
    # --------------------------------------------------------

    camera.set_focal_length(
        1.88
    )

    camera.set_horizontal_aperture(
        3.67
    )

    # Keep aperture ratio equal to 640:480.
    camera.set_vertical_aperture(
        3.67
        * CAMERA_RESOLUTION[1]
        / CAMERA_RESOLUTION[0]
    )

    carb.log_info(
        f"D435 camera position = "
        f"{CAMERA_POSITION.tolist()}"
    )

    carb.log_info(
        f"D435 target = "
        f"{CAMERA_TARGET.tolist()}"
    )

    return camera


# ============================================================
# Point D435 toward workspace
# ============================================================

def aim_d435():

    carb.log_info(
        "Aiming D435 at workspace center..."
    )

    # --------------------------------------------------------
    # Isaac Sim handles the camera orientation.
    #
    # This avoids manually mixing USD / ROS / optical-frame
    # axis conventions.
    # --------------------------------------------------------

    ViewportManager.set_camera_view(

        camera=(
            D435_CAMERA_PATH
        ),

        eye=(
            CAMERA_POSITION.tolist()
        ),

        target=(
            CAMERA_TARGET.tolist()
        ),
    )

    wait_updates(5)

    carb.log_info(
        "D435 orientation configured."
    )


# ============================================================
# Main Perspective camera
# ============================================================

def configure_main_view():

    carb.log_info(
        "Configuring main Perspective view..."
    )

    # --------------------------------------------------------
    # Main GUI camera.
    #
    # This is NOT D435.
    #
    # It exists only so we can inspect the entire scene.
    # --------------------------------------------------------

    try:

        ViewportManager.set_camera_view(

            camera=(
                "/OmniverseKit_Persp"
            ),

            eye=[
                1.60,
                -1.60,
                1.20,
            ],

            target=[
                0.35,
                0.0,
                0.15,
            ],
        )

        # Make sure active/default viewport uses Perspective.
        ViewportManager.set_camera(
            "/OmniverseKit_Persp"
        )

        carb.log_info(
            "Main Perspective view configured."
        )

    except Exception as error:

        carb.log_warn(
            "Main viewport configuration warning: "
            f"{error}"
        )


# ============================================================
# Create second D435 viewport
# ============================================================

def create_d435_viewport():

    carb.log_info(
        "========================================"
    )

    carb.log_info(
        "Creating D435 RGB Viewport"
    )

    carb.log_info(
        "========================================"
    )

    # --------------------------------------------------------
    # Isaac Sim 6.1 supports creating another viewport
    # directly for a specified camera.
    #
    # Main viewport stays Perspective.
    #
    # Second viewport becomes D435 RGB.
    # --------------------------------------------------------

    try:

        window = (
            ViewportManager
            .create_viewport_window(

                camera=(
                    D435_CAMERA_PATH
                ),

                title=(
                    "D435 RGB"
                ),

                resolution=(
                    CAMERA_RESOLUTION
                ),
            )
        )

        carb.log_info(
            "D435 RGB viewport created."
        )

        return window

    except Exception as error:

        carb.log_error(
            "Could not create D435 RGB viewport:"
        )

        carb.log_error(
            str(error)
        )

        return None


# ============================================================
# Camera data initialization
# ============================================================

def initialize_camera(
    camera,
    world,
):

    carb.log_info(
        "Initializing D435 RGB/Depth..."
    )

    # Give render products time to start.
    for _ in range(20):

        world.step(
            render=True
        )

    camera.initialize()

    # Depth output.
    camera.add_distance_to_image_plane_to_frame()

    # Give annotator more frames.
    for _ in range(30):

        world.step(
            render=True
        )

    carb.log_info(
        "D435 initialized."
    )


# ============================================================
# Live depth visualization
# ============================================================

def show_depth(depth):
    """Display a clipped, colorized depth map in a separate OpenCV window."""

    if cv2 is None:
        return False

    if depth is None:
        return False

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    valid = np.isfinite(depth) & (depth > 0.0)
    safe = np.full(depth.shape, DEPTH_MAX_M, dtype=np.float32)
    safe[valid] = depth[valid]
    safe = np.clip(safe, DEPTH_MIN_M, DEPTH_MAX_M)

    normalized = (safe - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
    depth_u8 = np.uint8(np.clip(normalized * 255.0, 0.0, 255.0))

    # Invert so near objects are visually stronger, then apply a colormap.
    depth_u8 = 255 - depth_u8
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

    # Mark invalid/infinite pixels black instead of allowing inf to corrupt scaling.
    depth_color[~valid] = (0, 0, 0)

    h, w = depth.shape[:2]
    center_v, center_u = h // 2, w // 2
    center_depth = depth[center_v, center_u]
    center_text = (
        f"center: {center_depth:.3f} m"
        if np.isfinite(center_depth) and center_depth > 0.0
        else "center: invalid"
    )

    cv2.putText(
        depth_color,
        f"Depth {DEPTH_MIN_M:.2f}-{DEPTH_MAX_M:.2f} m",
        (15, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        depth_color,
        center_text,
        (15, 55),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.drawMarker(
        depth_color,
        (center_u, center_v),
        (255, 255, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=18,
        thickness=1,
    )

    cv2.imshow(DEPTH_WINDOW_NAME, depth_color)
    cv2.waitKey(1)
    return True


# ============================================================
# Scene info
# ============================================================

def print_scene_info(
    robot_root
):

    carb.log_info(
        ""
    )

    carb.log_info(
        "========================================"
    )

    carb.log_info(
        "SCENE READY"
    )

    carb.log_info(
        "========================================"
    )

    carb.log_info(
        f"UR5e:"
    )

    carb.log_info(
        f"  {robot_root}"
    )

    carb.log_info(
        "Workspace:"
    )

    carb.log_info(
        "  /Worksurface"
    )

    carb.log_info(
        "D435 visible body:"
    )

    carb.log_info(
        "  /D435/Body"
    )

    carb.log_info(
        "D435 optical camera:"
    )

    carb.log_info(
        f"  {D435_CAMERA_PATH}"
    )

    carb.log_info(
        ""
    )

    carb.log_info(
        "Main viewport:"
    )

    carb.log_info(
        "  Perspective / world view"
    )

    carb.log_info(
        ""
    )

    carb.log_info(
        "Second viewport:"
    )

    carb.log_info(
        "  D435 RGB"
    )

    carb.log_info(
        ""
    )

    carb.log_info(
        f"D435 position:"
    )

    carb.log_info(
        f"  {CAMERA_POSITION.tolist()}"
    )

    carb.log_info(
        f"D435 target:"
    )

    carb.log_info(
        f"  {CAMERA_TARGET.tolist()}"
    )

    carb.log_info(
        "========================================"
    )


# ============================================================
# Main
# ============================================================

def main():

    try:

        # ----------------------------------------------------
        # 1. Import UR5e
        # ----------------------------------------------------

        robot_usd = (
            import_ur5e()
        )

        # ----------------------------------------------------
        # 2. Open imported USD directly
        # ----------------------------------------------------

        stage = (
            open_robot_stage(
                robot_usd
            )
        )
        # ----------------------------------------------------
        # 3. Robot root
        # ----------------------------------------------------

        robot_root = (
            find_robot_root(
                stage
            )
        )

        # ----------------------------------------------------
        # 4. World wrapper
        # ----------------------------------------------------

        world = World(
            stage_units_in_meters=1.0
        )

        # ----------------------------------------------------
        # 5. Ground plane (top surface = world Z 0)
        # ----------------------------------------------------

        create_ground(
            world,
            stage,
        )

        # ----------------------------------------------------
        # 6. Workspace
        # ----------------------------------------------------

        create_workspace(
            world,
            stage,
        )

        # ----------------------------------------------------
        # 6. Visible D435 body
        # ----------------------------------------------------

        create_d435_body(
            world,
            stage,
        )

        # ----------------------------------------------------
        # 7. Actual camera prim
        # ----------------------------------------------------

        camera = (
            create_d435()
        )

        # ----------------------------------------------------
        # 8. Render before camera manipulation
        # ----------------------------------------------------

        for _ in range(10):

            world.step(
                render=True
            )

        # ----------------------------------------------------
        # 9. Aim D435
        # ----------------------------------------------------

        aim_d435()

        # ----------------------------------------------------
        # 10. Initialize RGB / Depth
        # ----------------------------------------------------

        initialize_camera(
            camera,
            world,
        )

        # ----------------------------------------------------
        # 11. Configure world-view camera
        # ----------------------------------------------------

        configure_main_view()

        # ----------------------------------------------------
        # 12. Create separate D435 viewport
        # ----------------------------------------------------

        d435_viewport = (
            create_d435_viewport()
        )

        # Keep Python reference alive.
        _keep_d435_viewport_alive = (
            d435_viewport
        )

        # ----------------------------------------------------
        # 13. More frames
        # ----------------------------------------------------

        for _ in range(20):

            world.step(
                render=True
            )

        # ----------------------------------------------------
        # 14. Print configuration
        # ----------------------------------------------------

        print_scene_info(
            robot_root
        )

        # ----------------------------------------------------
        # 15. UR5e articulation + joint control GUI
        # ----------------------------------------------------

        # ----------------------------------------------------
        # IMPORTANT: build the PhysX simulation view BEFORE
        # SingleArticulation.initialize().  Without this reset,
        # Isaac Sim raises:
        # 'NoneType' object has no attribute 'create_articulation_view'
        # ----------------------------------------------------
        # Find the actual PhysX articulation root. Do not assume /ur5e.
        articulation_root = ensure_ur5e_articulation_root(
            stage,
            robot_hint=robot_root,
        )

        carb.log_info(
            f"Using UR5e articulation root: {articulation_root}"
        )

        # IMPORTANT: register the SingleArticulation with World BEFORE reset.
        # World.reset() is what creates the PhysX tensor articulation view.
        robot = create_ur5e_articulation(articulation_root, world)

        carb.log_info("Resetting world to build UR5e PhysX articulation view...")
        world.reset()

        # Let PhysX / articulation views settle for a few frames.
        for _ in range(10):
            world.step(render=True)

        robot, joint_indices, gripper_indices, gripper_multipliers, gripper_names = finalize_ur5e_articulation(robot)

        # Give the initialized articulation a few frames before the GUI starts.
        for _ in range(5):
            world.step(render=True)

        joint_gui = start_joint_control_gui()

        carb.log_info(
            "UR5e native Joint Control GUI started. "
            "The same enqueue_joint_command() path is reserved for U-Arm."
        )

        # ----------------------------------------------------
        # 16. Main loop
        # ----------------------------------------------------

        frame_counter = 0

        while (
            simulation_app.is_running()
        ):

            world.step(
                render=True
            )

            # Joint commands are executed only on Isaac Sim's main thread.
            process_joint_commands(
                robot,
                joint_indices,
            )
            process_gripper_commands(robot, gripper_indices, gripper_multipliers)

            frame_counter += 1

            # Update GUI current-angle labels at ~10 Hz.
            if frame_counter % 3 == 0:
                publish_joint_state(
                    robot,
                    joint_indices,
                    gripper_indices,
                    gripper_names,
                )
                joint_gui["refresh"]()

            # ------------------------------------------------
            # Live D435 depth visualization
            # ------------------------------------------------

            try:
                depth = camera.get_depth()

                if depth is not None:
                    show_depth(depth)

                # Log RGB/depth statistics about every 10 seconds.
                if frame_counter % 300 == 0:
                    rgba = camera.get_rgba()

                    if rgba is not None:
                        carb.log_info(f"D435 RGB: {rgba.shape}")
                    else:
                        carb.log_warn("D435 RGB is None.")

                    if depth is not None:
                        depth_np = np.asarray(depth, dtype=np.float32)
                        valid = np.isfinite(depth_np) & (depth_np > 0.0)
                        carb.log_info(f"D435 Depth: {depth_np.shape}")
                        if np.any(valid):
                            carb.log_info(
                                "D435 valid depth range: "
                                f"{depth_np[valid].min():.3f} - "
                                f"{depth_np[valid].max():.3f} m"
                            )
                    else:
                        carb.log_warn("D435 Depth is None.")

            except Exception as error:
                if frame_counter % 300 == 0:
                    carb.log_warn(
                        "D435 data/visualization warning: "
                        f"{error}"
                    )



    except Exception as error:

        carb.log_error(
            "========================================"
        )

        carb.log_error(
            "UR5e/D435 scene failed"
        )

        carb.log_error(
            "========================================"
        )

        carb.log_error(
            str(error)
        )

        raise

    finally:

        if cv2 is not None:
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass

        simulation_app.close()


# ============================================================
# Entry
# ============================================================

if __name__ == "__main__":

    main()