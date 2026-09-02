"""Direct URDF->USD conversion using the installed urdf importer extension API,
bypassing IsaacLab's UrdfConverter wrapper (incompatible with this Isaac Sim build)."""

import argparse
import os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("input", type=str)
parser.add_argument("output_dir", type=str)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import omni.kit.app

manager = omni.kit.app.get_app().get_extension_manager()
manager.set_extension_enabled_immediate("isaacsim.asset.importer.urdf", True)

from isaacsim.asset.importer.urdf import URDFImporter, URDFImporterConfig

urdf_path = os.path.abspath(args_cli.input)
usd_dir = os.path.abspath(args_cli.output_dir)  # a DIRECTORY - importer derives the filename itself
os.makedirs(usd_dir, exist_ok=True)

config = URDFImporterConfig(
    urdf_path=urdf_path,
    usd_path=usd_dir,
    merge_mesh=False,
    collision_from_visuals=False,
    collision_type="Convex Hull",
    allow_self_collision=False,
)

importer = URDFImporter(config)
output_path = importer.import_urdf()

print("-" * 80)
print(f"USD written to: {output_path}")
print("-" * 80)

simulation_app.close()