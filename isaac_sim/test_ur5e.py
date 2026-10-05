from isaacsim import SimulationApp

simulation_app = SimulationApp(
    {
        "headless": False,
    }
)

import carb
import omni.usd

from isaacsim.asset.importer.urdf import (
    URDFImporter,
    URDFImporterConfig,
)


URDF_PATH = "/tmp/ur5e_isaac.urdf"
USD_OUTPUT = "/tmp/ur5e_test.usd"


def main():

    carb.log_info("Creating URDF importer...")

    config = URDFImporterConfig(
        urdf_path=URDF_PATH,
        usd_path=USD_OUTPUT,

        fix_base=True,

        merge_fixed_joints=False,

        # 官方教學範例使用 merge_mesh=True
        merge_mesh=True,

        collision_from_visuals=False,

        allow_self_collision=False,

        robot_type="Manipulator",

        # 保留中間輸出，方便除錯
        debug_mode=True,
    )

    importer = URDFImporter(config)

    carb.log_info("Importing UR5e...")

    output_path = importer.import_urdf()

    carb.log_info(
        f"OUTPUT USD = {output_path}"
    )

    if not output_path:
        raise RuntimeError(
            "URDF import returned no USD."
        )

    # --------------------------------------------------------
    # IMPORTANT:
    # Open the importer output directly.
    # This follows Isaac Sim 6.1 official workflow.
    # --------------------------------------------------------

    carb.log_info(
        "Opening generated UR5e USD directly..."
    )

    omni.usd.get_context().open_stage(
        output_path
    )

    # Wait for USD/assets to load
    for _ in range(60):
        simulation_app.update()

    carb.log_info(
        "======================================"
    )

    carb.log_info(
        "UR5e USD OPENED"
    )

    carb.log_info(
        f"USD: {output_path}"
    )

    carb.log_info(
        "Select the robot in Stage and press F."
    )

    carb.log_info(
        "======================================"
    )

    while simulation_app.is_running():
        simulation_app.update()


try:

    main()

except Exception as error:

    carb.log_error(
        f"UR5e test failed: {error}"
    )

    raise

finally:

    simulation_app.close()
