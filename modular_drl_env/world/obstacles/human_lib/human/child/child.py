import os

import pybullet as p

from .. import Human


class Child(Human):
    def __init__(self,
                 pybtPhysicsClient,
                 partitioned=False,
                 self_collisions=False,
                 timestep=0.01,
                 scaling=1.0):
        """
        """
        if partitioned:
            print("TODO: Partitioned not done...")
            print("Loding un-partitioned urdf...")
            # self.body_id = p.loadURDF(
            # 	os.path.join(os.path.dirname(__file__),
            # 				 "child_partitioned.urdf"),
            # 	flags=p.URDF_MAINTAIN_LINK_ORDER,
            # 	physicsClientId=pybtPhysicsClient,
            # 	globalScaling=scaling
            # )
            urdf_load_flags = p.URDF_MAINTAIN_LINK_ORDER
            if self_collisions:
                urdf_load_flags = p.URDF_MAINTAIN_LINK_ORDER | p.URDF_USE_SELF_COLLISION
            self.body_id = p.loadURDF(
                os.path.join(os.path.dirname(__file__),
                             "child.urdf"),
                flags=urdf_load_flags,
                physicsClientId=pybtPhysicsClient,
                globalScaling=scaling
            )
        else:
            urdf_load_flags = p.URDF_MAINTAIN_LINK_ORDER
            if self_collisions:
                urdf_load_flags = p.URDF_MAINTAIN_LINK_ORDER | p.URDF_USE_SELF_COLLISION
            self.body_id = p.loadURDF(
                os.path.join(os.path.dirname(__file__),
                             "child.urdf"),
                flags=urdf_load_flags,
                physicsClientId=pybtPhysicsClient,
                globalScaling=scaling
            )

        super().__init__(
            pybtPhysicsClient,
            folder=os.path.dirname(__file__),
            timestep=timestep,
            scaling=scaling * 0.953/1.75,
            translation_scaling=0.95,   # this is a calibration/scaling of the mocap velocities
        )
