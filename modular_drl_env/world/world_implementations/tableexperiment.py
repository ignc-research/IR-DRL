from modular_drl_env.world.world import World
import numpy as np
import pybullet as pyb
from modular_drl_env.world.obstacles.human import Human
from modular_drl_env.world.obstacles.shapes import Box
import pybullet_data as pyb_d
from random import choice

__all__ = [
    'TableExperiment'
]

class TableExperiment(World):
    """
    Implements the table experiment with humans and moving obstacles by Kolja and Kai.
    """

    def __init__(self, workspace_boundaries: list, 
                       sim_step: float,
                       env_id: int,
                       num_obstacles: int,
                       obstacle_velocities: list,
                       num_humans: int,
                       human_positions: list,
                       human_rotations: list,
                       human_trajectories: list,
                       human_reactive: list,
                       ee_starts: list=[],
                       targets: list=[],
                       obstacle_positions: list=[],
                       obstacle_trajectories: list=[],
                       obstacle_training_schedule: bool=False):
        super().__init__(workspace_boundaries, sim_step, env_id)
        # INFO: if multiple robot base positions are given, we will assume that the first one is the main one for the experiment
        # also, we will always assume that the robot base is set up at 0,0,z
        # this will make generating obstacles easier

        # guard against usage with engines other than pybullet
        if self.engine.engine_type != "Pybullet":
            raise Exception("The table experiment cannot be used with engines other than Pybullet!")

        self.num_obstacles = num_obstacles
        self.num_humans = num_humans
        self.obstacle_velocities = obstacle_velocities

        # all of the following lists serve as overwrites for env functionality
        # useful for getting a repeatable starting point for evaluation
        # if left as empty lists the env will generate random ones, useful for training
        self.ee_starts = [np.array(ele) for ele in ee_starts]
        self.targets = [np.array(ele) for ele in targets]
        self.obstacle_positions = [np.array(ele) for ele in obstacle_positions]
        self.obstacle_trajectories = [[np.array(ele) for ele in traj] for traj in obstacle_trajectories]

        # table bounds, used for placing obstacles at random
        self.table_bounds_low = [-0.7, -0.7, 1.09]
        self.table_bounds_high = [0.7, 0.7, 1.85]

        # target bounds, used for placing the target, slightly tighter than the table bounds
        self.target_bounds_low = [-0.6, -0.6, 1.12]
        self.target_bounds_high = [0.6, 0.6, 1.35]  # bias towards being near the table, to keep it "realistic"

        # handle human stuff
        self.humans = []
        self.human_positions = [np.array(ele) for ele in human_positions]
        self.human_rotations = [np.array(ele) for ele in human_rotations]
        self.human_trajectories = [[np.array(ele) for ele in traj] for traj in human_trajectories]
        self.human_reactive = human_reactive  # list of bools that determines if the human in question will raise his arm if the robot gets near enough
        self.human_ee_was_near = [False for i in range(self.num_humans)]  # see update method
        self.near_threshold = 0.5

        self.obstacle_objects = []

        # wether num obstacles will be overwritten automatically depending on env success rate, might be useful for training
        self.obstacle_training_schedule = obstacle_training_schedule
        
    def build(self):
        # ground plate
        self.objects_ids.append(self.engine.add_ground_plane(np.array([0, 0, -0.01])))
        # table
        self.objects_ids.append(self.engine.load_urdf(pyb_d.getDataPath()+"/table/table.urdf", np.array([0, 0, 0]), np.array([0, 0, 0, 1]), scale=[1.75, 1.75, 1.75]))
        # humans
        for i in range(self.num_humans):
            human = Human(self.human_positions[i], self.human_rotations[i], self.human_trajectories[i], self.sim_step, 0.5, 1.5)
            human.build()
            self.humans.append(human)
        # obstacles
        extra = 0
        if np.random.random() < 0.3 and self.num_obstacles:  # generate a rather large brick moving about, this is a standard case that will appear in evaluation, useufl to have in training
            extra = 1
            idx = choice([0, 1])
            pos = [np.array([0, -0.45, 1.15]), np.array([0, 0.45, 1.15])]
            mov = 0.5 * self.sim_step
            traj = [[np.array([-0.6, -0.45, 1.15]), np.array([0.6, -0.45, 1.15])], [np.array([-0.6, 0.45, 1.15]), np.array([0.6, 0.45, 1.15])]]
            obs = Box(pos[idx], [0, 0, 0, 1], traj[idx], mov, [0.4, 0.1, 0.15], color=[0.75, 0, 0.25, 1])
            self.objects_ids.append(obs.build())
            self.obstacle_objects.append(obs)
        for i in range(self.num_obstacles - extra):
            # if there are no given obstacle positions, randomly generate some
            if not self.obstacle_positions:
                # first get the base position of the main robot, whcih we'll assume to be the first one
                base_position = self.robots_in_world[0].base_position
                # now we generate a position for the obstacle at random but while making sure that it doesn't spawn in a certain perimeter around the base
                while True:
                    position = np.random.uniform(low=self.table_bounds_low, high=self.table_bounds_high, size=(3,))
                    if np.linalg.norm(position - base_position) > 0.35:
                        break
            else:
                position = self.obstacle_positions[i]
            # if there are no given obstacle trajectories, randomly generate some
            if not self.obstacle_trajectories:
                trajectory = []
                if np.random.random() < 0.75:  # 25% will not move
                    trajectory_length = choice([2,3,4,5,6])
                    for i in range(trajectory_length):
                        while True:
                            # this creates positions for the trajectory that don't cross over the robot's own position
                            diff = position - self.robots_in_world[0].base_position
                            diff_norm = np.linalg.norm(diff)
                            point1 = self.robots_in_world[0].base_position + diff * (0.01 / diff_norm)
                            point2 = 2 * diff + self.robots_in_world[0].base_position
                            low = np.minimum(point1, point2)
                            high = np.maximum(point1, point2)
                            high[2] = 1.7
                            trajectory_element = np.random.uniform(low=low, high=high, size=(3,))
                            if np.linalg.norm(position - base_position) > 0.35:
                                trajectory.append(trajectory_element)
                                break
            else:
                trajectory = self.obstacle_trajectories[i]
            # if there are no given obstacle velocities, randomly generate some
            if not self.obstacle_velocities:
                move_step = np.random.uniform(low=0.01, high=0.5, size=(1,)) * self.sim_step
            else:
                move_step = self.obstacle_velocities[i] * self.sim_step
            # create random dimensions for obstacles 
            halfExtents = np.random.uniform(low=0.01, high=0.12, size=(3,)).tolist()
            # create somewhat random rotation
            #random_rpy = np.random.uniform(low=-np.pi/2, high=np.pi/2, size=(3,)).tolist()
            #random_quat = pyb.getQuaternionFromEuler(random_rpy)
            # create obstacles
            obs = Box(position, [0, 0, 0, 1], trajectory, move_step, halfExtents, color=[1, 0, 0, 1])      
            self.objects_ids.append(obs.build())
            self.obstacle_objects.append(obs)    

        # generate starting points and targets
        self._create_ee_starting_points(self.robots_in_world[0:1])
        min_dist = min((self.x_max - self.x_min) / 6, (self.y_max - self.y_min) / 6, (self.z_max - self.z_min) / 6)
        self._create_position_and_rotation_targets(self.robots_in_world[0:1], min_dist=min_dist)

        # move robots to starting position
        for idx, robot in enumerate(self.robots_in_world):
            if self.ee_starting_points[idx][0] is None:
                continue
            else:
                robot.moveto_joints(self.ee_starting_points[idx][2], False)

    def reset(self, success_rate):
        self.objects_ids = []
        self.position_targets = []
        self.rotation_targets = []
        self.ee_starting_points = []
        for human in self.humans:
            del human
        for obstacle in self.obstacle_objects:
            del obstacle
        self.obstacle_objects = []
        self.humans = []
        self.aux_object_ids = []

        if self.obstacle_training_schedule:
            if success_rate < 0.2:
                obs_mean = 0
            elif success_rate < 0.4:
                obs_mean = 1
            elif success_rate < 0.6:
                obs_mean = 2
            elif success_rate < 0.8:
                obs_mean = 3
            else:
                obs_mean = 5
            self.num_obstacles = round(np.random.normal(loc=obs_mean, scale=1.5))
            self.num_obstacles = min(8, self.num_obstacles)
            self.num_obstacles = max(0, self.num_obstacles)

    def update(self):
        for idx, human in enumerate(self.humans):
            human.move()
            if self.human_reactive[idx]:
                near = False or self.human_ee_was_near[idx]
                if not near:  # check if end effector is near
                    for robot in self.robots_in_world:
                        if np.linalg.norm(robot.position_rotation_sensor.position - human.position) <= self.near_threshold:
                            near = True
                            self.human_ee_was_near[idx] = True
                            break
                if near:
                    human.raise_hands()
        for obstacle in self.obstacle_objects:
            obstacle.move()

    def create_ee_starting_points(self) -> list:
        # use the preset starting points if there are some
        if self.ee_starts:
            ret = []
            for idx in range(len(self.ee_starts)):
                standard_rot = np.array([np.pi, 0, np.pi])
                random_rot = np.random.uniform(low=-np.pi, high=np.pi, size=(3,))
                standard_rot += random_rot * 0.1
                ret.append((self.ee_starts[idx], np.array(pyb.getQuaternionFromEuler(standard_rot.tolist()))))
                #ret.append((self.ee_start_overwrite[idx], None))
            return ret
        # otherwise, we simply put out nothing, making the robot start in its resting pose
        else:
            return [(None, None)]

    def create_position_target(self) -> list:
        # in contrast to other worlds, we will not check if for robots that need goals
        # this world only supports one robot with a position goal

        # use the preset targets if there are some
        if self.targets:
            self.position_targets = self.targets
            return self.targets
        # otherwise generate randomly
        else:
            while True:
                target = np.random.uniform(low=self.target_bounds_low, high=self.target_bounds_high, size=(3,))
                if np.linalg.norm(target - self.robots_in_world[0].base_position) > 0.15:
                    break
            self.position_targets = [target]
            return [target]

    def create_rotation_target(self) -> list:
        return None  # not needed for now