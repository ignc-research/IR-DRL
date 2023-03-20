from world.world import World
import numpy as np
import pybullet as pyb
from world.obstacles.human import Human
from world.obstacles.pybullet_shapes import Box
import pybullet_data as pyb_d
from random import choice
import sys
import os
from contextlib import contextmanager

__all__ = [
    'TableExperiment'
]
@contextmanager
def suppress_stdout():
    fd = sys.stdout.fileno()

    def _redirect_stdout(to):
        sys.stdout.close()  # + implicit flush()
        os.dup2(to.fileno(), fd)  # fd writes to 'to' file
        sys.stdout = os.fdopen(fd, "w")  # Python writes to fd

    with os.fdopen(os.dup(fd), "w") as old_stdout:
        with open(os.devnull, "w") as file:
            _redirect_stdout(to=file)
        try:
            yield  # allow code to be run with the redirected stdout
        finally:
            _redirect_stdout(to=old_stdout)  # restore stdout.
            # buffering and flags such as
            # CLOEXEC may be different

class TableExperiment(World):
    """
    Implements the table experiment with humans and moving obstacles by Kolja and Kai.
    """

    def __init__(self, world_config):
        super().__init__(world_config)
        # INFO: if multiple robot base positions are given, we will assume that the first one is the main one for the experiment
        # also, we will always assume that the robot base is set up at 0,0,z
        # this will make generating obstacle easier

        # if a experiment is given
        self.experiment = world_config["experiment"]

        self.num_obstacles = world_config["num_obstacles"]
        self.num_humans = world_config["num_humans"]
        self.obstacle_velocities = world_config["obstacle_velocities"]

        # all of the following lists serve as overwrites for env functionality
        # useful for getting a repeatable starting point for evaluation
        # if left as empty lists the env will generate random ones, useful for training
        self.ee_starts = [np.array(ele) for ele in world_config["ee_starts"]]
        self.targets = [np.array(ele) for ele in world_config["targets"]]
        self.obstacle_positions = [np.array(ele) for ele in world_config["obstacle_positions"]]
        self.obstacle_trajectories = [[np.array(ele) for ele in traj] for traj in world_config["obstacle_trajectories"]]

        # table bounds, used for placing obstacles at random
        self.table_bounds_low = [-0.7, -0.7, 1.09]
        self.table_bounds_high = [0.7, 0.7, 1.85]

        # target bounds, used for placing the target, slightly tighter than the table bounds
        self.target_bounds_low = [-0.6, -0.6, 1.12]
        self.target_bounds_high = [0.6, 0.6, 1.35]  # bias towards being near the table, to keep it "realistic"

        # handle human stuff
        self.humans = []
        self.human_positions = [np.array(ele) for ele in world_config["human_positions"]]
        self.human_rotations = [np.array(ele) for ele in world_config["human_rotations"]]
        self.human_trajectories = [[np.array(ele) for ele in traj] for traj in world_config["human_trajectories"]]
        self.human_reactive = world_config["human_reactive"]  # list of bools that determines if the human in question will raise his arm if the robot gets near enough
        self.human_ee_was_near = [False for i in range(self.num_humans)]  # see update method
        self.near_threshold = 0.5

        self.obstacle_objects = []

        # wether num obstacles will be overwritten automatically depending on env success rate, might be useful for training
        self.obstacle_training_schedule = world_config["obstacle_training_schedule"]

        # load targets
        if world_config["targets_path"] is not None:
            self.targets = np.loadtxt(world_config["targets_path"])
        
    def build(self):
        # ground plate
        self.objects_ids.append(pyb.loadURDF("workspace/plane.urdf", [0, 0, -0.01]))
        # table
        self.objects_ids.append(pyb.loadURDF(pyb_d.getDataPath()+"/table/table.urdf", useFixedBase=True, globalScaling=1.75))
        # humans
        for i in range(self.num_humans):
            with suppress_stdout():
                human = Human(self.human_positions[i], self.human_rotations[i], self.human_trajectories[i], self.sim_step, 1.5)
                human.build()
            self.humans.append(human)
        # obstacles
        extra = 0

        if self.experiment == 1:
            extra = 2
            idx = 0
            pos = [np.array([0, -0.45, 1.15]), np.array([0, 0.45, 1.15])]
            mov = 0.5 * self.sim_step
            traj = [[np.array([-0.6, -0.45, 1.15]), np.array([0.6, -0.45, 1.15])],
                    [np.array([-0.6, 0.45, 1.15]), np.array([0.6, 0.45, 1.15])]]
            obs = Box(pos[idx], [0, 0, 0, 1], traj[idx], mov, [0.4, 0.1, 0.15], color=[0.75, 0, 0.25, 1])
            print(idx)
            self.objects_ids.append(obs.build())
            self.obstacle_objects.append(obs)

        elif np.random.random() < 0.05 and self.num_obstacles:  # generate a rather large brick moving about, this is a standard case that will appear in evaluation, useufl to have in training
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
                # now we generate a position for the obstacle at random but while making sure that it doesn't spawn in a certain perimeter around the base and also the target
                while True:
                    position = np.random.uniform(low=self.table_bounds_low, high=self.table_bounds_high, size=(3,))
                    if np.linalg.norm(position - base_position) > 0.35 and np.linalg.norm(position - self.position_targets[0]) > 0.1:
                        break
            else:
                base_position = self.robots_in_world[0].base_position
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
                move_step = np.random.uniform(low=0.1, high=0.5, size=(1,)) * self.sim_step
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
        if self.experiment == 1:
            self.position_targets = [np.asarray([0.3,  -0.5,  1.2])]
            return [np.asarray([0,  1,  1.45])]
        if self.targets is not None:
            idx = np.random.randint(0, len(self.targets))
            self.position_targets = [self.targets[idx]]
            print([self.targets[idx]])
            return [self.targets[idx]]
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