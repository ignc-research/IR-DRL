import omni.ext
import omni.isaac.core.utils.prims as prims
import omni.isaac.core.utils.stage as stage

from omni.isaac.core.utils.nucleus import get_assets_root_path


# Any class derived from `omni.ext.IExt` in top level module (defined in `python.modules` of `extension.toml`) will be
# instantiated when extension gets enabled and `on_startup(ext_id)` will be called. Later when extension gets disabled
# on_shutdown() is called.
class IrIsaacBridgeExtension(omni.ext.IExt):
    # ext_id is current extension id. It can be used with extension manager to query additional information, like where
    # this extension is located on filesystem.
    def on_startup(self, ext_id):
        # get asset paths
        assets_root_path = get_assets_root_path()
        usd_path = assets_root_path + "/Isaac/Robots/Franka/franka.usd"

        # delete old franka, if exists
        prim_path = '/World/Franka'

        # old franka exists -> delete it
        if prims.find_matching_prim_paths(prim_path) != 0:
            prims.delete_prim(prim_path)
            print("Removing old franka instance")

        # add the Franka USD to our stage
        prims.create_prim(prim_path, prim_type="Xform")
        stage.add_reference_to_stage(usd_path, "/World/Franka")

    def on_shutdown(self):
        print("[ir.isaac.bridge] ir isaac bridge shutdown")
