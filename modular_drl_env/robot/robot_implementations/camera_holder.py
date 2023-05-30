from modular_drl_env.robot.robot import Robot

class CameraHolderUR5_Pybullet(Robot):

    def __init__(self, name: str, world, **kwargs):
        super().__init__(name, world, **kwargs)

        self.end_effector_link_id = "ee_link"
        self.base_link_id = "base_link"

        self.urdf_path = "ur5/urdf/ur5.urdf"