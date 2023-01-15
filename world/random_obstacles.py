from world.world import World
import numpy as np
import pybullet as pyb
from random import choice, shuffle

class RandomObstacleWorld(World):
    """
    This class generates a world with random box and sphere shaped obstacles.
    The obstacles will be placed between the p
    Depending on the configuration, some of these can be moving in various directions at various speeds
    """

    def __init__(self, workspace_boundaries: list=[-0.4, 0.4, 0.3, 0.7, 0.2, 0.5], 
                       robot_base_positions: list=[np.array([0.0, -0.12, 0.5])],
                       robot_base_orientations: list=[np.array([0, 0, 0, 1])],
                       num_static_obstacles: int=3, 
                       num_moving_obstacles: int=1,
                       box_measurements: list=[0.025, 0.075, 0.025, 0.075, 0.00075, 0.00125],
                       sphere_measurements: list=[0.005, 0.02],
                       moving_obstacles_vels: list=[0.0005, 0.005],
                       moving_obstacles_directions: list=[],
                       moving_obstacles_trajectory_length: list=[0.05, 0.75]
                       ):
        """
        :param workspace_boundaries: List of 6 floats containing the bounds of the workspace in the following order: xmin, xmax, ymin, ymax, zmin, zmax
        :param num_static_obstacles: int number that is the amount of static obstacles in the world
        :param num_moving_obstacles: int number that is the amount of moving obstacles in the world
        :param box_measurements: List of 6 floats that gives the minimum and maximum dimensions of box shapes in the following order: lmin, lmax, wmin, wmax, hmin, hmax
        :param sphere_measurements: List of 2 floats that gives the minimum and maximum radius of sphere shapes
        :param moving_obstacles_vels: List of 2 floats that gives the minimum and maximum velocity dynamic obstacles can move with
        :param moving_obstacles_directions: List of numpy arrays that contain directions in 3D space among which obstacles can move. If none are given directions are generated in random fashion.
        :param moving_obstacles_trajectory_length: List of 2 floats that contains the minimum and maximum trajectory length of dynamic obstacles.
        """

        # TODO: add random spline movement instead of simple linear trajectories
        # TODO: add random rotations for the plates

        super().__init__(workspace_boundaries, robot_base_positions, robot_base_orientations)

        self.num_static_obstacles = num_static_obstacles
        self.num_moving_obstacles = num_moving_obstacles

        self.box_l_min, self.box_l_max, self.box_w_min, self.box_w_max, self.box_h_min, self.box_h_max = box_measurements
        self.sphere_r_min, self.sphere_r_max = sphere_measurements

        self.vel_min, self.vel_max = moving_obstacles_vels
        self.vels = []  # to be populated in the build method

        self.allowed_directions = moving_obstacles_directions
        self.directions = []  # to be populated in the build method

        self.trajectory_length_min, self.trajectory_length_max = moving_obstacles_trajectory_length
        self.trajectory_lengths = [] # to be populated in the build method
        self.trajectory_state = []

        self.moving_obstacles_ids = []
        self.moving_obstacles_positions = []

    def build(self):

        # add ground plate
        ground_plate = pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01])
        self.objects_ids.append(ground_plate)

        # add the moving obstacles
        for i in range(self.num_moving_obstacles):
            # pick a one of the robots' starting positions randomly to...
            idx = choice(range(len(self.ee_starting_points)))
            # ... generate a random position between halfway between it and its target
            position = 0.5*(self.ee_starting_points[idx][0] + self.position_targets[idx] + 0.15*np.random.uniform(low=-1, high=1, size=(3,)))
            self.moving_obstacles_positions.append(position)
            
            # generate a velocity
            velocity = np.random.uniform(low=self.vel_min, high=self.vel_max)
            self.vels.append(velocity)

            # generate a trajectory length
            trajectory = np.random.uniform(low=self.trajectory_length_min, high=self.trajectory_length_max)
            self.trajectory_lengths.append(trajectory)
            self.trajectory_state.append(0)

            # get the direction from __init__ or, if none are given, generate one at random
            if self.allowed_directions:
                direction = self.allowed_directions[i]
            else:
                direction = np.random.uniform(low=-1, high=1, size=(3,))
            direction = (1 / np.linalg.norm(direction)) * direction
            self.directions.append(direction)
            
            # chance for plates 70%, for spheres 30%
            if np.random.random() > 0.3: 
                # plate
                plate = self._create_plate(position)
                self.moving_obstacles_ids.append(plate)
                self.objects_ids.append(plate)
            else:
                # sphere
                sphere = self._create_sphere(position)
                self.moving_obstacles_ids.append(sphere)
                self.objects_ids.append(sphere)

        # add the static obstacles
        for i in range(self.num_static_obstacles):
            idx = choice(range(len(self.ee_starting_points)))
            position = 0.5*(self.ee_starting_points[idx][0] + self.position_targets[idx] + 0.15*np.random.uniform(low=-1, high=1, size=(3,)))
            # chance for plates 70%, for spheres 30%
            if np.random.random() > 0.3: 
                # plate
                plate = self._create_plate(position)
                self.objects_ids.append(plate)
            else:
                # sphere
                sphere = self._create_sphere(position)
                self.objects_ids.append(sphere)

        # TODO: ensure that env does not result in instant collision

        self.built = True

    def reset(self):
        self.vels = []
        self.directions = []
        self.trajectory_lengths = []
        self.moving_obstacles_ids = []
        self.moving_obstacles_positions = []
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.ee_starting_points = []
        # the next three don't need to be reset, so commented out
        #self.robots_in_world = []
        #self.robots_with_position = []
        #self.robots_with_orientation = []

    def update(self):

        # TODO: safeguard against going out of bounds

        # move the moving obstacles according to their velocities, trajectory lengths and directions
        for idx, obstacle_id in enumerate(self.moving_obstacles_ids):
            trajectory_state = self.trajectory_state[idx]
            velocity = self.vels[idx]
            direction = self.directions[idx]
            max_trajectory = self.trajectory_lengths[idx]
            position = self.moving_obstacles_positions[idx]

            # first case: we're at least one step away from the end of the trajectory
            if trajectory_state + velocity < max_trajectory:
                new_position = velocity * direction + position
                self.trajectory_state[idx] += velocity
            # second case: one more step would go over the max trajectory
            else:
                temp_velocity = max_trajectory - trajectory_state
                new_position = temp_velocity * direction + position
                # after this, reverse the direction and set the trajectory state to 0
                self.directions[idx] = -self.directions[idx]
                self.trajectory_state[idx] = 0
            
            # apply movement
            pyb.resetBasePositionAndOrientation(obstacle_id, new_position, [0, 0, 0, 1])  # the quaternion was not specified when generating, so it's the default one here

    def _create_plate(self, position):
        length = np.random.uniform(low=self.box_l_min, high=self.box_l_max)
        width = np.random.uniform(low=self.box_w_min, high=self.box_w_max)
        height = np.random.uniform(low=self.box_h_min, high=self.box_h_max)

        # randomly assign lwh to xyz
        dims = [length, width, height]
        shuffle(dims)

        plate = pyb.createMultiBody(baseMass=0,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_BOX, halfExtents=[dims[0], dims[1], dims[2]], rgbaColor=[0.5,0.5,0.5,1]),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_BOX, halfExtents=[dims[0], dims[1], dims[2]]),
                                    basePosition=position)
        return plate

    def _create_sphere(self, position):
        radius = np.random.uniform(low=self.sphere_r_min, high=self.sphere_r_max)
        sphere = pyb.createMultiBody(baseMass=0,
                                    baseVisualShapeIndex=pyb.createVisualShape(shapeType=pyb.GEOM_SPHERE, radius=radius, rgbaColor=[0.75,0.75,0.75,1]),
                                    baseCollisionShapeIndex=pyb.createCollisionShape(shapeType=pyb.GEOM_SPHERE, radius=radius),
                                    basePosition=position)
        return sphere
        
    def create_ee_starting_points(self):
        for robot in self.robots_in_world:
            if robot in self.robots_with_position:
                rando = np.random.rand(3)
                x = (self.x_min + self.x_max) / 2 + 0.5 * (rando[0] - 0.5) * (self.x_max - self.x_min)
                y = (self.y_min + self.y_max) / 2 + 0.5 * (rando[1] - 0.5) * (self.y_max - self.y_min)
                z = (self.z_min + self.z_max) / 2 + 0.5 * (rando[2] - 0.5) * (self.z_max - self.z_min)
                self.ee_starting_points.append((np.array([x, y, z]), None))
            else:
                self.ee_starting_points.append([(None, None)])
        return self.ee_starting_points

    def create_position_target(self):
        for idx, robot in enumerate(self.robots_in_world):
            if robot in self.robots_with_position:
                while True:
                    rando = np.random.rand(3)
                    x = self.x_min + rando[0] * (self.x_max - self.x_min)
                    y = self.y_min + rando[1] * (self.y_max - self.y_min)
                    z = self.z_min + rando[2] * (self.z_max - self.z_min)
                    target = np.array([x, y, z])
                    if np.linalg.norm(target - self.ee_starting_points[idx][0]) > 0.4:
                        self.position_targets.append(target)
                        break
            else:
                self.position_targets.append([])
        return self.position_targets

    def create_rotation_target(self) -> list:
        return None  # TODO for later, just adding this so that the world starts

    def build_visual_aux(self):
        # create a visual border for the workspace
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                                lineToXYZ=[self.x_min, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])

        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_min, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_max],
                            lineToXYZ=[self.x_min, self.y_max, self.z_max])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_max],
                            lineToXYZ=[self.x_max, self.y_max, self.z_max])
        
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_min, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_max, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_min, self.y_min, self.z_min],
                            lineToXYZ=[self.x_min, self.y_max, self.z_min])
        pyb.addUserDebugLine(lineFromXYZ=[self.x_max, self.y_min, self.z_min],
                            lineToXYZ=[self.x_max, self.y_max, self.z_min])