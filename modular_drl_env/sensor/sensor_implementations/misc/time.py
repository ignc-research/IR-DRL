from gym.spaces import Box
import numpy as np
from modular_drl_env.sensor.sensor import Sensor

class TimeSensor(Sensor):
    """
    Sensor class that tracks either the amount of env steps or sim time and puts it into the observation space.
    """

    def __init__(self, 
                 normalize: bool, 
                 add_to_observation_space: bool, 
                 add_to_logging: bool, 
                 sim_step: float, 
                 sim_steps_per_env_step: int,
                 max_env_steps: int, 
                 report_steps: bool=True, 
                 report_time: bool=False):
        super().__init__(normalize, add_to_observation_space, add_to_logging, sim_step, 1, sim_steps_per_env_step)

        self.max_env_steps = max_env_steps
        self.max_time = self.max_env_steps * self.sim_step * self.sim_steps_per_env_step

        # flags for output
        self.report_steps = report_steps
        self.report_time = report_time

        if not report_steps and not report_time:
            raise Exception("Either time or steps must be reported by time sensor!")

        # data storage
        self.time = 0
        self.steps = 0

    def get_observation_space_element(self) -> dict:
        ret = dict()
        if self.report_time:
            if self.normalize:
                ret["time"] = Box(low=0, high=1, dtype=np.float32)
            else:
                ret["time"] = Box(low=0, high=self.max_time, dtype=np.float32)
        if self.report_steps:
            if self.normalize:
                ret["steps"] = Box(low=0, high=1, dtype=np.float32)
            else:
                ret["steps"] = Box(low=0, high=self.max_env_steps, dtype=np.int32)
        return ret
    
    def update(self, step) -> dict:
        self.steps = step
        self.time = self.sim_steps_per_env_step * self.sim_step * step
        return self.get_observation()

    def get_observation(self) -> dict:
        ret = dict()
        if self.report_time:
            if self.normalize:
                ret["time"] = self.time / self.max_time
            else:
                ret["time"] = self.time
        if self.report_steps:
            if self.normalize:
                ret["steps"] = self.steps / self.max_env_steps
            else:
                ret["steps"] = self.steps
        return ret
    
    def reset(self):
        self.steps = 0
        self.time = 0