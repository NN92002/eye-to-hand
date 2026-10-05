"""



Isaac Sim 6.1



UR5e + Robotiq 2F-140 + Workspace + D435 + LIVE Joint/Gripper GUI + ChArUco Eye-to-Hand (v20)







Main Viewport:



    Perspective view of entire scene







Second Viewport:



    D435 RGB camera view







Run:



    cd /home/neaf6000pro/isaacsim







    ./python.sh \\\\



    /home/n/eye-to-hand/isaac_sim/ur5e_camera_scene.py



"""


from pathlib import Path


import re


import numpy as np


import threading


import queue

import time

import csv

from datetime import datetime


# Optional OpenCV window for live depth visualization.


try:


    import cv2


except ImportError:


    cv2 = None


# ============================================================


# Start Isaac Sim


# ============================================================


STARTUP_STARTED_AT = time.perf_counter()
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


from pxr import Usd, UsdGeom, UsdPhysics, UsdShade, Sdf, Gf


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
def log_startup_checkpoint(label):
    elapsed = time.perf_counter() - STARTUP_STARTED_AT
    carb.log_info(f"[startup] {label}: total {elapsed:.2f} s")


def log_startup_duration(label, started_at):
    elapsed = time.perf_counter() - started_at
    carb.log_info(f"[startup] {label}: {elapsed:.2f} s")


log_startup_checkpoint("SimulationApp and Isaac imports ready")


# ============================================================


# Paths


# ============================================================


WORKSPACE_ROOT = Path(__file__).resolve().parent.parent


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

# Recording
RECORD_FPS = 30.0
RECORD_ROOT = Path(__file__).resolve().parent.parent / "recordings"
RECORD_COMMAND_QUEUE = queue.Queue()
RECORD_STATUS_QUEUE = queue.Queue(maxsize=1)


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
# UR5e base pose
# ============================================================

# Put the UR5e base at one corner of the 70 x 70 cm workspace.
# For the current workspace:
#   center = (0.45, 0.00)
#   selected corner = (0.10, -0.35)
#
# Z is the TOP surface of the workspace:
#   TABLE_HEIGHT + TABLE_THICKNESS = 0.03 m
#
# The base is rotated +45 degrees about world Z so it faces
# diagonally toward the workspace center.
UR5E_BASE_POSITION = np.array(
    [
        WORKSPACE_CENTER[0] - TABLE_WIDTH / 2.0,
        WORKSPACE_CENTER[1] - TABLE_DEPTH / 2.0,
        TABLE_HEIGHT + TABLE_THICKNESS,
    ],
    dtype=np.float32,
)

UR5E_BASE_YAW_DEG = 45.0


# ============================================================

# D435 pose

# ============================================================


# Environment camera placement in UR5e/world coordinates.

# Workspace center is at X=0.45 m. Put the camera 0.70 m in front

# of the workspace (-Y) and 0.90 m above the work plane.

# This moves the optical axis away from the robot body and gives

# a cleaner environment-camera view of the table.

CAMERA_POSITION = np.array(

    [

        0.90,   # X: aligned with workspace center

        0.0,   # Y: 70 cm in front of workspace

        0.9,   # Z: 90 cm above world/work plane

    ],

    dtype=np.float32,

)


# Look at the center of the work surface, slightly above the top face.

CAMERA_TARGET = np.array(

    [

        0.45,

        0.0,

        TABLE_HEIGHT + TABLE_THICKNESS,

    ],

    dtype=np.float32,

)


# ============================================================

# D435 camera settings

# ============================================================


CAMERA_RESOLUTION = (640, 480)

CAMERA_FREQUENCY = 30

CAMERA_NEAR_CLIP_M = 0.01

CAMERA_FAR_CLIP_M = 10.0

D435_CAMERA_PATH = "/D435/camera_color_optical_frame"


# ============================================================
# Eye-to-Hand ChArUco calibration (v20)
# ============================================================
# Fixed environment camera + ChArUco board rigidly attached to the robot tool.
# Convention used everywhere below:
#   T_A_B transforms a point expressed in frame B into frame A.

CHARUCO_SQUARES_X = 7
CHARUCO_SQUARES_Y = 5
CHARUCO_SQUARE_LENGTH_M = 0.040
CHARUCO_MARKER_LENGTH_M = 0.030
CHARUCO_MIN_CORNERS = 6
CHARUCO_MIN_SAMPLES = 10
CHARUCO_MAX_REPROJECTION_ERROR_PX = 1.5
CHARUCO_HAND_EYE_METHOD = "ANDREFF"
# Set from the camera's calibration file for real-camera use; None uses Isaac's intrinsics.
CHARUCO_CAMERA_MATRIX_OVERRIDE = None
# OpenCV distortion order: k1, k2, p1, p2, k3, ...; zeros model the distortion-free sim camera.
CHARUCO_DIST_COEFFS = np.zeros((5, 1), dtype=np.float64)
CHARUCO_DICTIONARY_ID = (
    cv2.aruco.DICT_5X5_250
    if cv2 is not None and hasattr(cv2, "aruco")
    else None
)

CHARUCO_BOARD_WIDTH_M = CHARUCO_SQUARES_X * CHARUCO_SQUARE_LENGTH_M
CHARUCO_BOARD_HEIGHT_M = CHARUCO_SQUARES_Y * CHARUCO_SQUARE_LENGTH_M
CHARUCO_TEXTURE_RESOLUTION = (1400, 1000)
CHARUCO_TEXTURE_PATH = WORKSPACE_ROOT / "isaac_sim" / "charuco_board.png"
CHARUCO_CALIBRATION_DIR = WORKSPACE_ROOT / "isaac_sim" / "calibration"
CHARUCO_WINDOW_NAME = "D435 ChArUco Eye-to-Hand"

# Board pose relative to the selected end-effector prim.
# The board plane is local XY and its front normal is +Z.
# If the board is not facing the D435 in your imported URDF, tune only these two
# constants; no calibration math needs to change.
CHARUCO_TOOL_TRANSLATION_M = np.array([0.0, 0.0, 0.12], dtype=np.float64)
CHARUCO_TOOL_RPY_DEG = np.array([0.0, 0.0, 0.0], dtype=np.float64)

CALIBRATION_COMMAND_QUEUE = queue.Queue()
CALIBRATION_STATUS_QUEUE = queue.Queue(maxsize=1)


# ============================================================

# D435 body visualization

# ============================================================


D435_BODY_SIZE = np.array(

    [0.090, 0.025, 0.025],

    dtype=np.float32,

)


# The old code put the opaque D435 body and the optical camera at

# exactly the same point, so the camera could render from inside its

# own cuboid. Keep the visible body behind the optical center instead.

D435_BODY_BACK_OFFSET_M = 0.06


def get_camera_view_direction():

    direction = np.asarray(CAMERA_TARGET - CAMERA_POSITION, dtype=np.float64)

    norm = float(np.linalg.norm(direction))

    if norm < 1e-9:

        raise RuntimeError("D435 camera position and target cannot be identical.")

    return (direction / norm).astype(np.float32)


def get_d435_body_position():

    return (

        CAMERA_POSITION

        - get_camera_view_direction() * D435_BODY_BACK_OFFSET_M

    ).astype(np.float32)


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


# ============================================================
# UR5e joint-angle conventions
# ============================================================
#
# IMPORTANT:
#   - UR5E_NATIVE_HOME_DEG is the pose that must be sent to the imported
#     Isaac Sim articulation to recover the original physically correct Home.
#   - UR5E_JOINT_DIRECTION maps the user/UR convention shown in the GUI to
#     Isaac Sim's imported joint convention.
#   - UR5E_HOME_DEG is therefore the GUI-space Home command, derived from
#     the native Isaac Home.  This keeps the physical Home pose unchanged
#     even when the GUI joint signs are reversed.
#
# Convention:
#     isaac_joint_deg = gui_joint_deg * UR5E_JOINT_DIRECTION
#
# If later testing shows that only some joints need inversion, change only
# UR5E_JOINT_DIRECTION; the Home pose will remain physically consistent.

UR5E_NATIVE_HOME_DEG = np.array(
    [0.0, -90.0, 90.0, -90.0, -90.0, 0.0],
    dtype=np.float32,
)


# GUI range. The imported URDF/robot controller remains the authority for


# the actual robot limits. This wide range is useful for simulation testing.


GUI_JOINT_MIN_DEG = -180.0


GUI_JOINT_MAX_DEG = 180.0


# Joint direction mapping between the user/UR convention shown in the GUI
# and the imported Isaac Sim articulation convention.
#
# Current TEST configuration: all six joints are inverted.
# Change individual entries to +1.0 after per-joint verification if needed.
UR5E_JOINT_DIRECTION = np.array(
    [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
    dtype=np.float32,
)

# GUI-space Home corresponding to the known-good Isaac-native Home above.
# With all directions = -1 this becomes:
#   [0, +90, -90, +90, +90, 0]
# and process_joint_commands() maps it back to the original physical Home:
#   [0, -90, +90, -90, -90, 0]
UR5E_HOME_DEG = (
    UR5E_NATIVE_HOME_DEG / UR5E_JOINT_DIRECTION
).astype(np.float32)


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


def enqueue_record_command(command):
    """Queue recording commands so file I/O stays on the Isaac Sim main thread."""
    if command not in ("start", "stop"):
        raise ValueError(f"Unknown record command: {command}")
    RECORD_COMMAND_QUEUE.put(command)


def push_record_status(text):
    try:
        while True:
            RECORD_STATUS_QUEUE.get_nowait()
    except queue.Empty:
        pass
    try:
        RECORD_STATUS_QUEUE.put_nowait(str(text))
    except queue.Full:
        pass


def enqueue_calibration_command(command):
    """Queue calibration actions; Isaac/USD/robot access remains on the main thread."""
    if command not in ("capture", "solve", "clear", "save"):
        raise ValueError(f"Unknown calibration command: {command}")
    CALIBRATION_COMMAND_QUEUE.put(command)


def push_calibration_status(status):
    """Publish the latest calibration status dict to the GUI without blocking."""
    try:
        while True:
            CALIBRATION_STATUS_QUEUE.get_nowait()
    except queue.Empty:
        pass
    try:
        CALIBRATION_STATUS_QUEUE.put_nowait(dict(status))
    except queue.Full:
        pass


def _matrix_from_rt(R, t):
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = np.asarray(R, dtype=np.float64).reshape(3, 3)
    T[:3, 3] = np.asarray(t, dtype=np.float64).reshape(3)
    return T


def _np_from_gf_matrix(matrix):
    return np.array([[float(matrix[r][c]) for c in range(4)] for r in range(4)], dtype=np.float64)


def get_prim_world_matrix(stage, prim_path):
    prim = stage.GetPrimAtPath(str(prim_path))
    if not prim.IsValid():
        raise RuntimeError(f"Invalid prim path for transform: {prim_path}")
    gf = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
    # Gf.Matrix4d stores affine transforms using row-vector convention. Transpose
    # so the returned NumPy matrix follows the column-vector convention used by
    # OpenCV and the rest of this file: p_A = T_A_B @ p_B.
    return _np_from_gf_matrix(gf).T


def find_end_effector_prim(stage, robot_root):
    """Prefer tool0; fall back to gripper/wrist links if the imported name differs."""
    priorities = [
        "tool0",
        "robotiq_arg2f_base_link",
        "robotiq_base_link",
        "flange",
        "wrist_3_link",
    ]
    root = robot_root.rstrip("/")
    candidates = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if path == root or path.startswith(root + "/"):
            candidates.append((prim.GetName().lower(), path))
    for wanted in priorities:
        for name, path in candidates:
            if name == wanted:
                carb.log_info(f"ChArUco parent end-effector prim: {path}")
                return path
    for wanted in priorities:
        for name, path in candidates:
            if wanted in name:
                carb.log_info(f"ChArUco parent end-effector prim (fuzzy): {path}")
                return path
    raise RuntimeError(
        "Could not find tool0/flange/wrist_3_link under the UR5e root. "
        "Inspect the imported prim hierarchy and add the correct link name to find_end_effector_prim()."
    )


def create_charuco_board_definition():
    if cv2 is None or not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "OpenCV ArUco module is unavailable. Isaac's Python environment needs an OpenCV build with cv2.aruco."
        )
    dictionary = cv2.aruco.getPredefinedDictionary(CHARUCO_DICTIONARY_ID)
    board = cv2.aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        float(CHARUCO_SQUARE_LENGTH_M),
        float(CHARUCO_MARKER_LENGTH_M),
        dictionary,
    )
    return dictionary, board


def generate_charuco_texture(board):
    CHARUCO_TEXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    width, height = CHARUCO_TEXTURE_RESOLUTION
    if hasattr(board, "generateImage"):
        image = board.generateImage((int(width), int(height)), marginSize=20, borderBits=1)
    else:
        image = np.zeros((int(height), int(width)), dtype=np.uint8)
        board.draw((int(width), int(height)), image, marginSize=20, borderBits=1)
    if not cv2.imwrite(str(CHARUCO_TEXTURE_PATH), image):
        raise RuntimeError(f"Could not write ChArUco texture: {CHARUCO_TEXTURE_PATH}")
    carb.log_info(f"ChArUco texture generated: {CHARUCO_TEXTURE_PATH}")


def create_charuco_board_prim(stage, end_effector_path):
    """Create a textured, visual-only calibration board as a child of the tool link."""
    board_root_path = end_effector_path.rstrip("/") + "/CharucoCalibrationTarget"
    old = stage.GetPrimAtPath(board_root_path)
    if old.IsValid():
        stage.RemovePrim(board_root_path)

    root = UsdGeom.Xform.Define(stage, board_root_path)
    xform_api = UsdGeom.XformCommonAPI(root.GetPrim())
    xform_api.SetTranslate(Gf.Vec3d(*[float(v) for v in CHARUCO_TOOL_TRANSLATION_M]))
    xform_api.SetRotate(
        Gf.Vec3f(*[float(v) for v in CHARUCO_TOOL_RPY_DEG]),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )

    mesh_path = board_root_path + "/BoardMesh"
    mesh = UsdGeom.Mesh.Define(stage, mesh_path)
    w = float(CHARUCO_BOARD_WIDTH_M)
    h = float(CHARUCO_BOARD_HEIGHT_M)
    mesh.CreatePointsAttr([
        Gf.Vec3f(-w / 2.0, -h / 2.0, 0.0),
        Gf.Vec3f(+w / 2.0, -h / 2.0, 0.0),
        Gf.Vec3f(+w / 2.0, +h / 2.0, 0.0),
        Gf.Vec3f(-w / 2.0, +h / 2.0, 0.0),
    ])
    mesh.CreateFaceVertexCountsAttr([4])
    mesh.CreateFaceVertexIndicesAttr([0, 1, 2, 3])
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeom.Tokens.none)
    mesh.CreateDoubleSidedAttr(True)

    primvars = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    st = primvars.CreatePrimvar(
        "st",
        Sdf.ValueTypeNames.TexCoord2fArray,
        UsdGeom.Tokens.vertex,
    )
    st.Set([
        Gf.Vec2f(0.0, 1.0),
        Gf.Vec2f(1.0, 1.0),
        Gf.Vec2f(1.0, 0.0),
        Gf.Vec2f(0.0, 0.0),
    ])

    material_path = board_root_path + "/Material"
    material = UsdShade.Material.Define(stage, material_path)
    shader = UsdShade.Shader.Define(stage, material_path + "/PreviewSurface")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.6)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)

    texture = UsdShade.Shader.Define(stage, material_path + "/Texture")
    texture.CreateIdAttr("UsdUVTexture")
    texture.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(str(CHARUCO_TEXTURE_PATH)))
    texture.CreateInput("sourceColorSpace", Sdf.ValueTypeNames.Token).Set("sRGB")

    reader = UsdShade.Shader.Define(stage, material_path + "/PrimvarReader")
    reader.CreateIdAttr("UsdPrimvarReader_float2")
    reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    texture.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.ConnectableAPI(), "result")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(texture.ConnectableAPI(), "rgb")
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(material)

    carb.log_info(
        f"ChArUco board created at {board_root_path} | "
        f"size={w:.3f} x {h:.3f} m | local XYZ={CHARUCO_TOOL_TRANSLATION_M.tolist()} | "
        f"local RPY={CHARUCO_TOOL_RPY_DEG.tolist()} deg"
    )
    return board_root_path


def get_camera_intrinsics(camera):
    """Use an explicit override, Isaac intrinsics, or the configured camera model."""
    if CHARUCO_CAMERA_MATRIX_OVERRIDE is not None:
        matrix = np.asarray(CHARUCO_CAMERA_MATRIX_OVERRIDE, dtype=np.float64)
        if (
            matrix.shape != (3, 3)
            or not np.all(np.isfinite(matrix))
            or matrix[0, 0] <= 0.0
            or matrix[1, 1] <= 0.0
        ):
            raise ValueError("CHARUCO_CAMERA_MATRIX_OVERRIDE must be a finite 3x3 matrix.")
        return matrix, "explicit override"

    try:
        K = np.asarray(camera.get_intrinsics_matrix(), dtype=np.float64)
        if (K.shape == (3, 3) and np.all(np.isfinite(K))
                and K[0, 0] > 0.0 and K[1, 1] > 0.0):
            return K, "Isaac camera API"
    except Exception as error:
        carb.log_warn(f"camera.get_intrinsics_matrix() unavailable, using analytic fallback: {error!r}")

    width, height = [float(v) for v in CAMERA_RESOLUTION]
    focal_mm = 1.88
    horizontal_aperture_mm = 3.67
    vertical_aperture_mm = 3.67 * height / width
    fx = focal_mm / horizontal_aperture_mm * width
    fy = focal_mm / vertical_aperture_mm * height
    cx = (width - 1.0) / 2.0
    cy = (height - 1.0) / 2.0
    matrix = np.array(
        [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return matrix, "configured analytic fallback"


def rotation_error_deg(R_est, R_gt):
    R_delta = np.asarray(R_gt, dtype=np.float64).T @ np.asarray(R_est, dtype=np.float64)
    value = (np.trace(R_delta) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(value, -1.0, 1.0))))


def quaternion_wxyz_to_matrix(q):
    q = np.asarray(q, dtype=np.float64).reshape(4)
    w, x, y, z = q
    n = np.linalg.norm(q)
    if n < 1e-12:
        raise RuntimeError("Zero-length quaternion")
    w, x, y, z = q / n
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)],
    ], dtype=np.float64)


class EyeToHandCalibrator:
    """Main-thread ChArUco detector/sample collector/OpenCV eye-to-hand solver."""

    def __init__(self, stage, robot_base_path, tool_path, camera, dictionary, board):
        self.stage = stage
        self.robot_base_path = robot_base_path
        self.tool_path = tool_path
        self.camera = camera
        self.dictionary = dictionary
        self.board = board
        self.camera_matrix, self.camera_matrix_source = get_camera_intrinsics(camera)
        self.dist_coeffs = np.asarray(CHARUCO_DIST_COEFFS, dtype=np.float64).reshape(-1, 1)
        if (
            self.dist_coeffs.size not in (4, 5, 8, 12, 14)
            or not np.all(np.isfinite(self.dist_coeffs))
        ):
            raise ValueError("CHARUCO_DIST_COEFFS must contain 4, 5, 8, 12, or 14 finite values.")
        self.samples = []
        self.latest_T_camera_target = None
        self.latest_corner_count = 0
        self.latest_reprojection_error = None
        self.latest_annotated_bgr = None
        self.last_result = None
        self.last_status_text = "Ready - move robot so the ChArUco board is visible"

        if hasattr(cv2.aruco, "CharucoDetector"):
            self.detector = cv2.aruco.CharucoDetector(self.board)
        else:
            self.detector = None

        carb.log_info(
            "Eye-to-Hand camera intrinsic matrix:\n"
            + np.array2string(self.camera_matrix, precision=5)
        )
        self.publish_status(False)

    def publish_status(self, detected):
        status = {
            "detected": bool(detected),
            "corners": int(self.latest_corner_count),
            "samples": int(len(self.samples)),
            "reproj_px": None if self.latest_reprojection_error is None else float(self.latest_reprojection_error),
            "text": str(self.last_status_text),
        }
        if self.last_result is not None:
            status.update({
                "translation_error_mm": self.last_result.get("translation_error_mm"),
                "rotation_error_deg": self.last_result.get("rotation_error_deg"),
                "method": self.last_result.get("method"),
                "method_summary": self.last_result.get("method_summary", ""),
                "method_diagnostics": self.last_result.get("method_diagnostics", {}),
            })
        else:
            status["method"] = CHARUCO_HAND_EYE_METHOD
            status["method_summary"] = ""
        push_calibration_status(status)

    def update_detection(self, rgba, show_window=True):
        self.latest_T_camera_target = None
        self.latest_corner_count = 0
        self.latest_reprojection_error = None
        if rgba is None:
            self.publish_status(False)
            return False

        rgb = np.asarray(rgba)
        if rgb.ndim != 3 or rgb.shape[-1] < 3:
            self.publish_status(False)
            return False
        rgb = rgb[..., :3].astype(np.uint8, copy=False)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        if self.detector is not None:
            charuco_corners, charuco_ids, marker_corners, marker_ids = self.detector.detectBoard(gray)
        else:
            marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, self.dictionary)
            if marker_ids is None:
                charuco_corners, charuco_ids = None, None
            else:
                _, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
                    marker_corners,
                    marker_ids,
                    gray,
                    self.board,
                    cameraMatrix=self.camera_matrix,
                    distCoeffs=self.dist_coeffs,
                )

        detected = charuco_ids is not None and charuco_corners is not None
        if not detected:
            self.latest_annotated_bgr = bgr
            if show_window:
                self._show_annotated()
            self.publish_status(False)
            return False

        ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
        corners = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
        self.latest_corner_count = int(len(ids))
        if len(ids) < CHARUCO_MIN_CORNERS:
            self.latest_annotated_bgr = bgr
            if show_window:
                self._show_annotated()
            self.publish_status(False)
            return False

        all_obj = np.asarray(self.board.getChessboardCorners(), dtype=np.float64).reshape(-1, 3)
        obj_points = all_obj[ids]
        image_points = corners

        ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            image_points,
            self.camera_matrix,
            self.dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            self.latest_annotated_bgr = bgr
            if show_window:
                self._show_annotated()
            self.publish_status(False)
            return False

        R, _ = cv2.Rodrigues(rvec)
        self.latest_T_camera_target = _matrix_from_rt(R, tvec)
        projected, _ = cv2.projectPoints(
            obj_points,
            rvec,
            tvec,
            self.camera_matrix,
            self.dist_coeffs,
        )
        projected = projected.reshape(-1, 2)
        self.latest_reprojection_error = float(np.mean(np.linalg.norm(projected - image_points, axis=1)))

        try:
            cv2.aruco.drawDetectedCornersCharuco(
                bgr,
                np.asarray(charuco_corners, dtype=np.float32),
                np.asarray(charuco_ids, dtype=np.int32),
            )
            cv2.drawFrameAxes(
                bgr,
                self.camera_matrix,
                self.dist_coeffs,
                rvec,
                tvec,
                0.08,
                2,
            )
        except Exception:
            pass

        cv2.putText(
            bgr,
            f"corners={self.latest_corner_count} reproj={self.latest_reprojection_error:.2f}px samples={len(self.samples)}",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
        self.latest_annotated_bgr = bgr
        if show_window:
            self._show_annotated()
        self.publish_status(True)
        return True

    def _show_annotated(self):
        try:
            cv2.imshow(CHARUCO_WINDOW_NAME, self.latest_annotated_bgr)
            cv2.waitKey(1)
        except Exception as error:
            carb.log_warn(f"ChArUco window warning: {error!r}")

    def current_T_base_tool(self):
        T_world_base = get_prim_world_matrix(self.stage, self.robot_base_path)
        T_world_tool = get_prim_world_matrix(self.stage, self.tool_path)
        return np.linalg.inv(T_world_base) @ T_world_tool

    def ground_truth_T_base_camera(self):
        """Use Isaac camera's ROS optical-frame world pose so GT matches OpenCV solvePnP axes."""
        try:
            position, orientation = self.camera.get_world_pose(camera_axes="ros")
            T_world_camera = _matrix_from_rt(
                quaternion_wxyz_to_matrix(orientation),
                position,
            )
            T_world_base = get_prim_world_matrix(self.stage, self.robot_base_path)
            return np.linalg.inv(T_world_base) @ T_world_camera
        except Exception as error:
            carb.log_warn(f"Could not compute camera ground truth with camera_axes='ros': {error!r}")
            return None

    def capture_sample(self):
        if self.latest_T_camera_target is None:
            self.last_status_text = "CAPTURE rejected: ChArUco pose is not currently valid"
            self.publish_status(False)
            return
        if (
            self.latest_reprojection_error is None
            or self.latest_reprojection_error > CHARUCO_MAX_REPROJECTION_ERROR_PX
        ):
            self.last_status_text = (
                "CAPTURE rejected: reprojection error "
                f"{self.latest_reprojection_error:.2f}px > {CHARUCO_MAX_REPROJECTION_ERROR_PX:.2f}px"
            )
            self.publish_status(True)
            return

        T_base_tool = self.current_T_base_tool()
        self.samples.append(
            {
                "T_base_tool": T_base_tool.copy(),
                "T_camera_target": self.latest_T_camera_target.copy(),
                "reprojection_error_px": float(self.latest_reprojection_error),
                "corner_count": int(self.latest_corner_count),
            }
        )
        self.last_result = None
        self.last_status_text = f"Captured sample {len(self.samples)}"
        carb.log_info(
            f"Eye-to-Hand sample {len(self.samples)} captured | "
            f"corners={self.latest_corner_count} | reproj={self.latest_reprojection_error:.3f}px"
        )
        self.publish_status(True)

    def clear_samples(self):
        self.samples.clear()
        self.last_result = None
        self.last_status_text = "Calibration samples cleared"
        self.publish_status(self.latest_T_camera_target is not None)

    def solve(self):
        if len(self.samples) < CHARUCO_MIN_SAMPLES:
            self.last_status_text = (
                f"Need at least {CHARUCO_MIN_SAMPLES} samples; currently {len(self.samples)}"
            )
            self.publish_status(self.latest_T_camera_target is not None)
            return None

        method_ids = {
            "PARK": cv2.CALIB_HAND_EYE_PARK,
            "ANDREFF": cv2.CALIB_HAND_EYE_ANDREFF,
        }
        selected_method = CHARUCO_HAND_EYE_METHOD.upper()
        if selected_method not in method_ids:
            self.last_status_text = (
                f"Unknown hand-eye method {CHARUCO_HAND_EYE_METHOD!r}; choose PARK or ANDREFF"
            )
            self.publish_status(self.latest_T_camera_target is not None)
            return None

        # OpenCV calibrateHandEye() is documented using T_base_gripper as input and
        # returning T_gripper_camera for eye-in-hand. For eye-to-hand, invert the
        # robot poses to T_tool_base and pass those in the first argument slot;
        # the returned transform then corresponds to T_base_camera.
        R_base2tool = []
        t_base2tool = []
        R_target2cam = []
        t_target2cam = []
        for sample in self.samples:
            T_tool_base = np.linalg.inv(sample["T_base_tool"])
            T_camera_target = sample["T_camera_target"]
            R_base2tool.append(T_tool_base[:3, :3].copy())
            t_base2tool.append(T_tool_base[:3, 3].reshape(3, 1).copy())
            R_target2cam.append(T_camera_target[:3, :3].copy())
            t_target2cam.append(T_camera_target[:3, 3].reshape(3, 1).copy())

        T_gt = self.ground_truth_T_base_camera()
        method_estimates = {}
        method_diagnostics = {}
        for method_name, method_id in method_ids.items():
            try:
                R_base_camera, t_base_camera = cv2.calibrateHandEye(
                    R_base2tool,
                    t_base2tool,
                    R_target2cam,
                    t_target2cam,
                    method=method_id,
                )
                estimate = _matrix_from_rt(R_base_camera, t_base_camera)
                if not np.all(np.isfinite(estimate)):
                    raise ValueError("solver returned non-finite values")
                method_estimates[method_name] = estimate
                method_diagnostics[method_name] = self._evaluate_estimate(estimate, T_gt)
            except Exception as error:
                method_diagnostics[method_name] = {"error": str(error)}

        if selected_method not in method_estimates:
            self.last_status_text = f"Selected solver {selected_method} failed"
            carb.log_error(
                f"Eye-to-Hand solver {selected_method} failed: "
                f"{method_diagnostics[selected_method].get('error', 'unknown error')}"
            )
            self.publish_status(self.latest_T_camera_target is not None)
            return None

        T_base_camera = method_estimates[selected_method]
        selected_diagnostics = method_diagnostics[selected_method]
        method_summary = self._format_method_summary(method_diagnostics)

        result = {
            "method": selected_method,
            "samples": len(self.samples),
            "T_base_camera": T_base_camera,
            "T_base_camera_GT": T_gt,
            "method_estimates": method_estimates,
            "method_diagnostics": method_diagnostics,
            "method_summary": method_summary,
            "translation_error_mm": selected_diagnostics.get("translation_error_mm"),
            "rotation_error_deg": selected_diagnostics.get("rotation_error_deg"),
        }

        self.last_result = result
        self.last_status_text = f"Calibration solved with {selected_method}"
        carb.log_info("========== EYE-TO-HAND CALIBRATION ==========")
        carb.log_info(f"Samples: {len(self.samples)}")
        carb.log_info(f"Selected method: {selected_method}")
        carb.log_info(
            "Estimated T_base_camera:\n"
            + np.array2string(T_base_camera, precision=7, suppress_small=True)
        )
        for method_name, diagnostics in method_diagnostics.items():
            carb.log_info(f"{method_name} diagnostics: {diagnostics}")
        if T_gt is not None:
            carb.log_info(
                "Isaac Sim ground-truth T_base_camera:\n"
                + np.array2string(T_gt, precision=7, suppress_small=True)
            )
        carb.log_info("=============================================")
        self.publish_status(self.latest_T_camera_target is not None)
        return result

    def _evaluate_estimate(self, T_base_camera, T_gt):
        implied_tool_targets = [
            np.linalg.inv(sample["T_base_tool"])
            @ T_base_camera
            @ sample["T_camera_target"]
            for sample in self.samples
        ]
        translations = np.stack([transform[:3, 3] for transform in implied_tool_targets])
        translation_center = np.mean(translations, axis=0)
        translation_rms_mm = float(
            np.sqrt(np.mean(np.sum((translations - translation_center) ** 2, axis=1)))
            * 1000.0
        )

        rotations = np.stack([transform[:3, :3] for transform in implied_tool_targets])
        left, _, right = np.linalg.svd(np.mean(rotations, axis=0))
        correction = np.eye(3)
        correction[2, 2] = np.linalg.det(left @ right)
        mean_rotation = left @ correction @ right
        rotation_errors = np.asarray(
            [rotation_error_deg(rotation, mean_rotation) for rotation in rotations]
        )

        diagnostics = {
            "closure_translation_rms_mm": translation_rms_mm,
            "closure_rotation_rms_deg": float(np.sqrt(np.mean(rotation_errors ** 2))),
        }
        if T_gt is not None:
            diagnostics["translation_error_mm"] = float(
                np.linalg.norm(T_base_camera[:3, 3] - T_gt[:3, 3]) * 1000.0
            )
            diagnostics["rotation_error_deg"] = rotation_error_deg(
                T_base_camera[:3, :3], T_gt[:3, :3]
            )
        return diagnostics

    @staticmethod
    def _format_method_summary(method_diagnostics):
        parts = []
        for method_name, diagnostics in method_diagnostics.items():
            if "error" in diagnostics:
                parts.append(f"{method_name}: failed")
                continue
            if "translation_error_mm" in diagnostics:
                quality = f"GT {diagnostics['translation_error_mm']:.2f} mm"
            else:
                quality = "GT n/a"
            parts.append(
                f"{method_name}: {quality}, fit "
                f"{diagnostics['closure_translation_rms_mm']:.2f} mm"
            )
        return " | ".join(parts)

    def save(self):
        if self.last_result is None:
            self.last_status_text = "SAVE rejected: solve the calibration first"
            self.publish_status(self.latest_T_camera_target is not None)
            return
        CHARUCO_CALIBRATION_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        stem = CHARUCO_CALIBRATION_DIR / f"eye_to_hand_{stamp}"
        np.save(str(stem) + "_T_base_camera.npy", self.last_result["T_base_camera"])
        if self.last_result.get("T_base_camera_GT") is not None:
            np.save(str(stem) + "_T_base_camera_GT.npy", self.last_result["T_base_camera_GT"])
        np.savez(
            str(stem) + "_samples.npz",
            T_base_tool=np.stack([s["T_base_tool"] for s in self.samples], axis=0),
            T_camera_target=np.stack([s["T_camera_target"] for s in self.samples], axis=0),
            reprojection_error_px=np.asarray([s["reprojection_error_px"] for s in self.samples]),
            corner_count=np.asarray([s["corner_count"] for s in self.samples]),
            camera_matrix=self.camera_matrix,
            dist_coeffs=self.dist_coeffs,
            camera_resolution=np.asarray(CAMERA_RESOLUTION, dtype=np.int32),
            opencv_version=np.asarray(cv2.__version__),
            method_names=np.asarray(list(self.last_result["method_estimates"])),
            T_base_camera_methods=np.stack(
                list(self.last_result["method_estimates"].values()), axis=0
            ),
        )

        yaml_path = Path(str(stem) + ".yaml")
        T = self.last_result["T_base_camera"]
        with open(yaml_path, "w") as f:
            f.write("convention: 'T_A_B maps coordinates from frame B to frame A'\n")
            f.write("parent_frame: base\n")
            f.write("child_frame: camera_color_optical_frame\n")
            f.write(f"method: {self.last_result['method']}\n")
            f.write(f"samples: {self.last_result['samples']}\n")
            f.write(f"camera_matrix_source: {self.camera_matrix_source}\n")
            f.write(f"opencv_version: {cv2.__version__}\n")
            f.write(f"camera_resolution: {CAMERA_RESOLUTION[0]} x {CAMERA_RESOLUTION[1]}\n")
            f.write("camera_matrix:\n")
            for row in self.camera_matrix:
                f.write("  - [" + ", ".join(f"{float(v):.10f}" for v in row) + "]\n")
            distortion_values = ", ".join(
                f"{float(value):.10f}" for value in self.dist_coeffs.reshape(-1)
            )
            f.write(f"dist_coeffs: [{distortion_values}]\n")
            f.write("T_base_camera:\n")
            for row in T:
                f.write("  - [" + ", ".join(f"{float(v):.10f}" for v in row) + "]\n")
            f.write("method_diagnostics:\n")
            for method_name, diagnostics in self.last_result["method_diagnostics"].items():
                f.write(f"  {method_name}:\n")
                for key, value in diagnostics.items():
                    f.write(f"    {key}: {value}\n")
            if self.last_result.get("translation_error_mm") is not None:
                f.write(
                    "translation_error_mm: "
                    f"{self.last_result['translation_error_mm']:.6f}\n"
                )
                f.write(
                    "rotation_error_deg: "
                    f"{self.last_result['rotation_error_deg']:.6f}\n"
                )

        self.last_status_text = f"Saved: {yaml_path.name}"
        carb.log_info(f"Eye-to-Hand result saved under: {stem}")
        self.publish_status(self.latest_T_camera_target is not None)

    def process_commands(self):
        actions = {
            "capture": self.capture_sample,
            "solve": self.solve,
            "clear": self.clear_samples,
            "save": self.save,
        }
        while True:
            try:
                command = CALIBRATION_COMMAND_QUEUE.get_nowait()
            except queue.Empty:
                break
            if command in actions:
                actions[command]()


class D435Recorder:
    """Record synchronized RGB MP4, depth heatmap MP4, and per-frame depth NPY."""

    def __init__(self, fps=RECORD_FPS):
        self.fps = float(fps)
        self.active = False
        self.output_dir = None
        self.depth_dir = None
        self.rgb_writer = None
        self.heat_writer = None
        self.csv_file = None
        self.csv_writer = None
        self.frame_index = 0
        self.next_capture_wall_time = None

    def start(self):
        if self.active:
            push_record_status(f"RECORDING: {self.output_dir}")
            return
        if cv2 is None:
            raise RuntimeError("OpenCV is unavailable; cannot record MP4 files.")

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = RECORD_ROOT / stamp
        self.depth_dir = self.output_dir / "depth_npy"
        self.depth_dir.mkdir(parents=True, exist_ok=True)

        width, height = int(CAMERA_RESOLUTION[0]), int(CAMERA_RESOLUTION[1])
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self.rgb_writer = cv2.VideoWriter(
            str(self.output_dir / "rgb.mp4"), fourcc, self.fps, (width, height)
        )
        self.heat_writer = cv2.VideoWriter(
            str(self.output_dir / "depth_heatmap.mp4"), fourcc, self.fps, (width, height)
        )
        if not self.rgb_writer.isOpened():
            raise RuntimeError("Could not open RGB MP4 writer.")
        if not self.heat_writer.isOpened():
            raise RuntimeError("Could not open depth heatmap MP4 writer.")

        self.csv_file = open(self.output_dir / "timestamps.csv", "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow([
            "frame_id",
            "wall_time_s",
            "depth_file",
            "rgb_video",
            "heatmap_video",
        ])

        with open(self.output_dir / "recording_info.txt", "w") as f:
            f.write(f"resolution={width}x{height}\n")
            f.write(f"fps={self.fps}\n")
            f.write(f"depth_unit=meter\n")
            f.write(f"depth_dtype=float32\n")
            f.write(f"depth_min_visualization_m={DEPTH_MIN_M}\n")
            f.write(f"depth_max_visualization_m={DEPTH_MAX_M}\n")

        self.frame_index = 0
        self.next_capture_wall_time = time.perf_counter()
        self.active = True
        carb.log_info(f"D435 recording STARTED: {self.output_dir}")
        push_record_status(f"RECORDING: {self.output_dir.name}")

    def stop(self):
        if self.rgb_writer is not None:
            self.rgb_writer.release()
            self.rgb_writer = None
        if self.heat_writer is not None:
            self.heat_writer.release()
            self.heat_writer = None
        if self.csv_file is not None:
            self.csv_file.flush()
            self.csv_file.close()
            self.csv_file = None
            self.csv_writer = None

        if self.active:
            carb.log_info(
                f"D435 recording STOPPED: {self.output_dir} | frames={self.frame_index}"
            )
            push_record_status(
                f"SAVED: {self.output_dir.name} ({self.frame_index} frames)"
            )
        self.active = False
        self.next_capture_wall_time = None

    def should_capture_now(self):
        if not self.active:
            return False
        now = time.perf_counter()
        if self.next_capture_wall_time is None:
            self.next_capture_wall_time = now
        if now + 1e-9 < self.next_capture_wall_time:
            return False
        period = 1.0 / self.fps
        while self.next_capture_wall_time <= now:
            self.next_capture_wall_time += period
        return True

    def write_frame(self, rgba, depth, heatmap_bgr):
        if not self.active or not self.should_capture_now():
            return
        if rgba is None or depth is None or heatmap_bgr is None:
            return

        rgba = np.asarray(rgba)
        depth = np.asarray(depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[..., 0]

        # Isaac camera RGBA -> OpenCV BGR.
        if rgba.ndim == 3 and rgba.shape[-1] >= 3:
            rgb = rgba[..., :3].astype(np.uint8, copy=False)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        else:
            return

        expected_w, expected_h = int(CAMERA_RESOLUTION[0]), int(CAMERA_RESOLUTION[1])
        if (bgr.shape[1], bgr.shape[0]) != (expected_w, expected_h):
            bgr = cv2.resize(bgr, (expected_w, expected_h), interpolation=cv2.INTER_AREA)
        if (heatmap_bgr.shape[1], heatmap_bgr.shape[0]) != (expected_w, expected_h):
            heatmap_bgr = cv2.resize(
                heatmap_bgr, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST
            )

        depth_name = f"depth_{self.frame_index:06d}.npy"
        depth_path = self.depth_dir / depth_name
        np.save(depth_path, depth.astype(np.float32, copy=False))

        self.rgb_writer.write(bgr)
        self.heat_writer.write(heatmap_bgr)
        self.csv_writer.writerow([
            self.frame_index,
            f"{time.time():.6f}",
            f"depth_npy/{depth_name}",
            "rgb.mp4",
            "depth_heatmap.mp4",
        ])
        if self.frame_index % 30 == 0:
            self.csv_file.flush()

        self.frame_index += 1
        push_record_status(
            f"RECORDING: {self.output_dir.name} | frames={self.frame_index}"
        )


def process_record_commands(recorder):
    newest = None
    while True:
        try:
            newest = RECORD_COMMAND_QUEUE.get_nowait()
        except queue.Empty:
            break
    if newest == "start":
        recorder.start()
    elif newest == "stop":
        recorder.stop()


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


    state = {"latest": None, "gripper_latest": None, "suppress_live_send": False, "suppress_gripper_send": False, "record_status": "IDLE"}


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

            carb.log_info(
                "UR5e Home requested | GUI deg="
                + np.array2string(UR5E_HOME_DEG, precision=1)
                + " | Isaac native deg="
                + np.array2string(UR5E_NATIVE_HOME_DEG, precision=1)
            )

            status_label.text = "Home target sent (physical native Home preserved)"


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


    def start_recording():
        enqueue_record_command("start")
        status_label.text = "Recording start requested..."

    def stop_recording():
        enqueue_record_command("stop")
        status_label.text = "Recording stop requested..."


    def capture_calibration_sample():
        enqueue_calibration_command("capture")
        status_label.text = "Calibration capture requested..."

    def solve_calibration():
        enqueue_calibration_command("solve")
        status_label.text = "Calibration solve requested..."

    def clear_calibration_samples():
        enqueue_calibration_command("clear")
        status_label.text = "Calibration clear requested..."

    def save_calibration_result():
        enqueue_calibration_command("save")
        status_label.text = "Calibration save requested..."

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


            ui.Separator(height=4)
            ui.Label("D435 Recording", height=24, alignment=ui.Alignment.CENTER)
            with ui.HStack(height=34, spacing=10):
                ui.Spacer()
                ui.Button("START RECORDING", width=170, clicked_fn=start_recording)
                ui.Button("STOP RECORDING", width=170, clicked_fn=stop_recording)
                ui.Spacer()
            recording_status_label = ui.Label(
                "Recording: IDLE",
                height=22,
                alignment=ui.Alignment.CENTER,
            )


            ui.Separator(height=4)
            ui.Label("Eye-to-Hand ChArUco Calibration", height=24, alignment=ui.Alignment.CENTER)
            calibration_detect_label = ui.Label(
                "ChArUco: waiting...",
                height=20,
                alignment=ui.Alignment.CENTER,
            )
            calibration_sample_label = ui.Label(
                "Samples: 0",
                height=20,
                alignment=ui.Alignment.CENTER,
            )
            calibration_error_label = ui.Label(
                "Isaac Sim GT error: --",
                height=20,
                alignment=ui.Alignment.CENTER,
            )
            calibration_solver_label = ui.Label(
                f"Solver: {CHARUCO_HAND_EYE_METHOD}",
                height=20,
                alignment=ui.Alignment.CENTER,
            )
            calibration_method_labels = {
                method_name: ui.Label(
                    f"{method_name}: --",
                    height=20,
                    alignment=ui.Alignment.CENTER,
                )
                for method_name in ("PARK", "ANDREFF")
            }
            with ui.HStack(height=34, spacing=6):
                ui.Button("CAPTURE", width=120, clicked_fn=capture_calibration_sample)
                ui.Button("SOLVE", width=100, clicked_fn=solve_calibration)
                ui.Button("CLEAR", width=100, clicked_fn=clear_calibration_samples)
                ui.Button("SAVE", width=100, clicked_fn=save_calibration_result)

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

        record_status = None
        try:
            while True:
                record_status = RECORD_STATUS_QUEUE.get_nowait()
        except queue.Empty:
            pass
        if record_status is not None:
            state["record_status"] = record_status
            recording_status_label.text = record_status


        calibration_status = None
        try:
            while True:
                calibration_status = CALIBRATION_STATUS_QUEUE.get_nowait()
        except queue.Empty:
            pass
        if calibration_status is not None:
            detected = calibration_status.get("detected", False)
            corners = calibration_status.get("corners", 0)
            reproj = calibration_status.get("reproj_px")
            samples = calibration_status.get("samples", 0)
            if reproj is None:
                reproj_text = "--"
            else:
                reproj_text = f"{float(reproj):.2f}px"
            calibration_detect_label.text = (
                f"ChArUco: {'DETECTED' if detected else 'NOT DETECTED'} | "
                f"corners={corners} | reproj={reproj_text}"
            )
            calibration_sample_label.text = (
                f"Samples: {samples} | {calibration_status.get('text', '')}"
            )
            selected_method = calibration_status.get("method", CHARUCO_HAND_EYE_METHOD)
            calibration_solver_label.text = f"Solver: {selected_method}"
            method_diagnostics = calibration_status.get("method_diagnostics", {})
            for method_name, label in calibration_method_labels.items():
                diagnostics = method_diagnostics.get(method_name)
                if not diagnostics:
                    label.text = f"{method_name}: --"
                elif "error" in diagnostics:
                    label.text = f"{method_name}: failed"
                else:
                    fit_error = diagnostics["closure_translation_rms_mm"]
                    gt_error = diagnostics.get("translation_error_mm")
                    if gt_error is None:
                        label.text = f"{method_name}: fit {fit_error:.2f} mm"
                    else:
                        label.text = (
                            f"{method_name}: Sim GT {gt_error:.2f} mm | "
                            f"fit {fit_error:.2f} mm"
                        )
            t_err = calibration_status.get("translation_error_mm")
            r_err = calibration_status.get("rotation_error_deg")
            if t_err is not None and r_err is not None:
                calibration_error_label.text = (
                    "Isaac Sim GT error: "
                    f"{float(t_err):.3f} mm | {float(r_err):.4f} deg"
                )
            else:
                calibration_error_label.text = "Isaac Sim GT error: unavailable"

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


    push_latest_state(np.rad2deg(current[arm_indices]) * UR5E_JOINT_DIRECTION)


    main_slot = gripper_names.index(ROBOTIQ_MAIN_JOINT)


    push_latest_gripper_state(current[gripper_indices[main_slot]])


    carb.log_info("Initial UR5e + Robotiq articulation state is finite.")


    return robot, arm_indices, gripper_indices, gripper_multipliers, gripper_names


def process_joint_commands(robot, joint_indices):


    """



    Apply pending six-axis commands from the GUI (and later U-Arm).







    The GUI/U-Arm layer uses the user/UR sign convention in degrees.
    Mapping to the imported Isaac joint convention and conversion to radians
    happen only here.



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


    # GUI/U-Arm targets use the desired UR5e sign convention. The imported
    # Isaac articulation rotates all six arm joints in the opposite direction,
    # so map the commanded angles before converting to radians.
    isaac_target_deg = (
        np.asarray(target_deg, dtype=np.float32)
        * UR5E_JOINT_DIRECTION
    )
    target_rad = np.deg2rad(isaac_target_deg).astype(np.float32)


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
        + " deg -> Isaac "
        + np.array2string(isaac_target_deg, precision=1)


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


    push_latest_state(np.rad2deg(current[joint_indices]) * UR5E_JOINT_DIRECTION)


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


    importer_started_at = time.perf_counter()
    source_urdf = SOURCE_URDF


    if not source_urdf.exists():


        raise FileNotFoundError(


            f"Expanded URDF not found: {source_urdf}\n"


            "Generate it first with ROS2 xacro."


        )


    mesh_root = UR_DESCRIPTION_DIR


    if not mesh_root.exists():


        raise FileNotFoundError(


            f"Universal Robot source directory not found: {mesh_root}"


        )


    # The expanded URDF still contains package://ur_description/...


    # Resolve those URLs so the standalone Isaac Sim process does not need ROS.


    resolved_urdf = WORKSPACE_ROOT / "isaac_sim" / "ur5e_robotiq_2f140_isaac_resolved.urdf"


    urdf_text = source_urdf.read_text()


    urdf_text = urdf_text.replace(


        "package://ur_description/",


        mesh_root.as_posix().rstrip("/") + "/",


    )


    robotiq_mesh_root = ROBOTIQ_DESCRIPTION_DIR


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


    output_dir = WORKSPACE_ROOT / "isaac_sim" / "generated_ur5e_robotiq_2f140_nomimic_v19_camera_fix"


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


    urdf_conversion_started_at = time.perf_counter()
    output_path = importer.import_urdf()
    log_startup_duration("URDFImporter.import_urdf", urdf_conversion_started_at)


    if not output_path:


        raise RuntimeError("URDFImporter returned an empty USD output path.")


    output_path = str(output_path)


    carb.log_info(f"UR5e USD generated: {output_path}")


    # Keep compatibility with the scene's existing startup flow:


    # main() opens the generated robot USD as the active stage, then adds


    # workspace, ground and D435 into that same stage.


    carb.log_info(f"UR5e USD ready to open: {output_path}")
    log_startup_duration("URDF preparation and import", importer_started_at)


    return output_path


def open_robot_stage(


    usd_path


):


    stage_open_started_at = time.perf_counter()
    carb.log_info(


        "Opening UR5e USD..."


    )


    omni.usd.get_context().open_stage(


        usd_path


    )


    # Give USD / materials time to load.


    wait_updates(60)
    log_startup_duration("open robot stage and 60 USD updates", stage_open_started_at)


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


def set_ur5e_base_pose(stage, robot_root):
    """Place the UR5e base at a workspace corner and rotate it toward the center."""

    if robot_root is None:
        raise RuntimeError("UR5e root prim is None.")

    prim = stage.GetPrimAtPath(robot_root)

    if not prim.IsValid():
        raise RuntimeError(f"Invalid UR5e root prim: {robot_root}")

    carb.log_info("========================================")
    carb.log_info("Setting UR5e base pose")
    carb.log_info("========================================")

    xform_api = UsdGeom.XformCommonAPI(prim)

    xform_api.SetTranslate(
        Gf.Vec3d(
            float(UR5E_BASE_POSITION[0]),
            float(UR5E_BASE_POSITION[1]),
            float(UR5E_BASE_POSITION[2]),
        )
    )

    xform_api.SetRotate(
        Gf.Vec3f(
            0.0,
            0.0,
            float(UR5E_BASE_YAW_DEG),
        ),
        UsdGeom.XformCommonAPI.RotationOrderXYZ,
    )

    carb.log_info(
        f"UR5e base position = {UR5E_BASE_POSITION.tolist()}"
    )
    carb.log_info(
        f"UR5e base yaw = {UR5E_BASE_YAW_DEG:.1f} deg"
    )


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


    # Workspace center marker intentionally disabled in calibration mode.

    # Keep the tabletop visually clean so only the robot-mounted ChArUco

    # target is used for calibration.


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

    carb.log_info("Creating visible D435 body...")


    body_path = "/D435/Body"

    prim = stage.GetPrimAtPath(body_path)

    if prim.IsValid():

        stage.RemovePrim(body_path)


    # Keep the opaque body behind the optical center. This prevents

    # the RGB camera from starting inside the D435 visualization mesh.

    body_position = get_d435_body_position()


    body = FixedCuboid(

        prim_path=body_path,

        name="d435_body",

        position=body_position,

        scale=D435_BODY_SIZE.copy(),

        color=np.array([0.15, 0.15, 0.15], dtype=np.float32),

    )

    world.scene.add(body)


    carb.log_info(f"D435 optical center = {CAMERA_POSITION.tolist()}")

    carb.log_info(f"D435 visible body position = {body_position.tolist()}")


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


def configure_d435_clipping():

    """Set the USD camera clipping range after the camera prim exists."""

    stage = omni.usd.get_context().get_stage()

    if stage is None:

        raise RuntimeError("USD stage is unavailable while configuring D435 clipping.")


    prim = stage.GetPrimAtPath(D435_CAMERA_PATH)

    if not prim.IsValid():

        raise RuntimeError(f"D435 camera prim not found: {D435_CAMERA_PATH}")


    usd_camera = UsdGeom.Camera(prim)

    usd_camera.GetClippingRangeAttr().Set(

        Gf.Vec2f(float(CAMERA_NEAR_CLIP_M), float(CAMERA_FAR_CLIP_M))

    )

    carb.log_info(

        f"D435 clipping range = {CAMERA_NEAR_CLIP_M:.3f} - "

        f"{CAMERA_FAR_CLIP_M:.1f} m"

    )


# ============================================================


# Camera data initialization


# ============================================================


def initialize_camera(


    camera,


    world,


):


    camera_init_started_at = time.perf_counter()
    carb.log_info(


        "Initializing D435 RGB/Depth..."


    )


    # Give render products time to start.


    for _ in range(20):


        world.step(


            render=True


        )


    camera.initialize()


    configure_d435_clipping()


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
    log_startup_duration("camera init and 50 render frames", camera_init_started_at)


# ============================================================


# Live depth visualization


# ============================================================


def initialize_depth_window():
    """Explicitly create the third live depth-heatmap window."""
    if cv2 is None:
        carb.log_warn("OpenCV import failed; D435 depth heatmap window is unavailable.")
        return False
    try:
        cv2.namedWindow(DEPTH_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(
            DEPTH_WINDOW_NAME,
            int(CAMERA_RESOLUTION[0]),
            int(CAMERA_RESOLUTION[1]),
        )
        cv2.moveWindow(DEPTH_WINDOW_NAME, 60, 60)
        try:
            cv2.startWindowThread()
        except Exception:
            pass
        carb.log_info("D435 depth heatmap window created.")
        return True
    except Exception as error:
        carb.log_error(f"Could not create D435 depth heatmap window: {error!r}")
        return False


def make_depth_heatmap(depth):
    """Convert float32 depth in meters into a BGR TURBO heatmap."""
    if cv2 is None or depth is None:
        return None

    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]

    valid = np.isfinite(depth) & (depth > 0.0)
    safe = np.full(depth.shape, DEPTH_MAX_M, dtype=np.float32)
    safe[valid] = depth[valid]
    safe = np.clip(safe, DEPTH_MIN_M, DEPTH_MAX_M)

    normalized = (safe - DEPTH_MIN_M) / (DEPTH_MAX_M - DEPTH_MIN_M)
    depth_u8 = np.uint8(np.clip(normalized * 255.0, 0.0, 255.0))
    depth_u8 = 255 - depth_u8
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)
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
    return depth_color


def show_depth(depth):
    """Display the live depth heatmap and return the BGR heatmap frame."""
    depth_color = make_depth_heatmap(depth)
    if depth_color is None:
        return None
    try:
        cv2.imshow(DEPTH_WINDOW_NAME, depth_color)
        cv2.waitKey(1)
    except Exception as error:
        carb.log_error(f"D435 depth heatmap display failed: {error!r}")
        return None
    return depth_color


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
        log_startup_checkpoint("URDF ready")


        # ----------------------------------------------------


        # 2. Open imported USD directly


        # ----------------------------------------------------


        stage = (


            open_robot_stage(


                robot_usd


            )


        )
        log_startup_checkpoint("robot USD stage ready")


        # ----------------------------------------------------


        # 3. Robot root


        # ----------------------------------------------------


        robot_root = (


            find_robot_root(


                stage


            )


        )


        # ----------------------------------------------------
        # 3.1 Place UR5e at workspace corner, Z on tabletop
        # ----------------------------------------------------

        set_ur5e_base_pose(
            stage,
            robot_root,
        )


        # ----------------------------------------------------
        # 3.2 ChArUco target attached to the end effector
        # ----------------------------------------------------
        charuco_dictionary, charuco_board = create_charuco_board_definition()
        generate_charuco_texture(charuco_board)
        charuco_tool_path = find_end_effector_prim(stage, robot_root)
        charuco_board_path = create_charuco_board_prim(stage, charuco_tool_path)
        carb.log_info(f"ChArUco target prim = {charuco_board_path}")


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
        log_startup_checkpoint("camera ready")


        initialize_depth_window()


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
        physics_init_started_at = time.perf_counter()


        world.reset()


        # Let PhysX / articulation views settle for a few frames.


        for _ in range(10):


            world.step(render=True)
        log_startup_duration("world reset and 10 settling render frames", physics_init_started_at)


        robot, joint_indices, gripper_indices, gripper_multipliers, gripper_names = finalize_ur5e_articulation(robot)


        # Give the initialized articulation a few frames before the GUI starts.


        for _ in range(5):


            world.step(render=True)


        eye_to_hand = EyeToHandCalibrator(
            stage=stage,
            robot_base_path=robot_root,
            tool_path=charuco_tool_path,
            camera=camera,
            dictionary=charuco_dictionary,
            board=charuco_board,
        )

        joint_gui = start_joint_control_gui()
        log_startup_checkpoint("control GUI ready")


        carb.log_info(


            "UR5e native Joint Control GUI started. "


            "The same enqueue_joint_command() path is reserved for U-Arm."


        )


        # ----------------------------------------------------


        # 16. Main loop


        # ----------------------------------------------------


        recorder = D435Recorder(fps=RECORD_FPS)
        push_record_status("Recording: IDLE")

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
            process_record_commands(recorder)
            eye_to_hand.process_commands()


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
                rgba = camera.get_rgba()
                heatmap_bgr = None

                if depth is not None:
                    heatmap_bgr = show_depth(depth)

                if rgba is not None:
                    eye_to_hand.update_detection(rgba, show_window=True)

                if recorder.active:
                    recorder.write_frame(rgba, depth, heatmap_bgr)

                # Log RGB/depth statistics about every 10 seconds.
                if frame_counter % 300 == 0:
                    if rgba is not None:
                        carb.log_info(f"D435 RGB: {np.asarray(rgba).shape}")
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

        try:
            if "eye_to_hand" in locals() and hasattr(eye_to_hand, "close"):
                eye_to_hand.close()
        except Exception as error:
            carb.log_warn(f"Calibrator cleanup warning: {error!r}")

        try:
            if "recorder" in locals():
                recorder.stop()
        except Exception as error:
            carb.log_warn(f"Recorder cleanup warning: {error!r}")

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