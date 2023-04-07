from abc import ABC, abstractmethod
from time import time
from modular_drl_env.engine.engine import get_instance

class Sensor(ABC):
    """
    Abstract Base Class for a sensor. The methods signed with abstractmethod have to be implemented in subclasses.
    See the joint or position sensor for examples.
    """

    def __init__(self, normalize: bool, add_to_observation_space: bool, add_to_logging: bool, sim_step: float, update_steps: int, sim_steps_per_env_step: int):
        
        super().__init__()
        
        # get engine
        self.engine = get_instance()

        # determines whether the output of the sensor is normalized to be between -1 and 1 (or alternatively between 0 and 1, if that makes more sense for a particular type of sensor)
        # note: class variables as well as the logging output should still be unnormalized, only the output of get_data() should be changed by this
        self.normalize = normalize
        
        # determines wether this sensor will create a field for the observation space, default is yes
        # useful to set to false if you need the sensor data for something (e.g. logging) but don't want the model to have access to it
        self.add_to_observation_space = add_to_observation_space

        # determines wether a sensor will output logging data
        # useful to set to false if you need the sensor data for your model but don't want it in the logs
        self.add_to_logging = add_to_logging

        # time that passes per sim step
        self.sim_step = sim_step
        self.sim_steps_per_env_step = sim_steps_per_env_step

        # use these two variables to determine relative CPU (aka real-world) time, useful for performance measuring
        self.cpu_time = 0
        self.cpu_epoch = time()

        # set the update rate
        # use this value to set the rate at which the sensor will update its data, useful to conserve fps
        # see the joint sensor implementation for an example 
        self.update_steps = update_steps

        # list of auxillary visual objects, this gets purged every env step!
        self.aux_visual_objects = []

    @abstractmethod
    def update(self, step) -> dict:
        """
        Updates sensor data by performing underlying data collection.
        This also includes any operations necessary to adapt the sensor's state to the changed environment, e.g. in case of moving sensors.
        The method receives the current env step from outside, might be useful for certain things.
        Returns the current data.
        """
        pass

    @abstractmethod
    def reset(self):
        """
        This should set any values within the sensor back to default.
        Should be used when tracking e.g. previous position such that velocities are tracked correctly at the start of an episode.
        This should also get a new epoch by calling time() and reset self.time to 0 (see already implemented sensors).
        """
        pass

    @abstractmethod
    def get_observation(self) -> dict:
        """
        Returns the data currently stored. Does not perform an update.
        This must return the data in the same format as defined below in the gym space.
        """
        pass

    @abstractmethod
    def _normalize(self) -> dict:
        """
        Returns the sensor data in normalized format.
        """
        pass

    @abstractmethod
    def get_observation_space_element(self) -> dict:
        """
        Returns a dict with gym spaces to be used as a part of the observation space of the parent gym env. Called once at the init of the gym env.
        Dict keys should contain a sensible name for the data and the name of the robot, values should be associated gym spaces.
        Example for position sensor: "position_ur5_1": gym.space.Box(low=0, high=1, shape=(6,), dtype=np.float32)
        """
        pass

    def get_data_for_logging(self) -> dict:
        """
        This method can be used to return data in a format that is more useful for logging.
        Besides the primary data returned by get_observation(), this can also include secondary data (e.g. the position sensor does not return absolute position in get_data, so this is a way to make that data available).
        Output should be a dict with parameter name as key and the value as value.
        If there is no use case for this with a specific sensor implementation, return an empty dict instead.
        You can also add custom methods for returning specific pieces of data other than this method, however only this method will be used for automatic logging.
        """
        return {}

    def build_visual_aux(self):
        """
        You can use this method to draw visual objects useful for demonstration or debug purposes.
        Add all these objects' ids to self.aux_visual_pyb_objects, where they will get deleted from on every step automatically. 
        """
        pass

    def delete_visual_aux(self):
        """
        Deletes all visual aides created by this sensor.
        """
        for aux_object in self.aux_visual_objects:
            self.engine.remove_aux_object(aux_object)
        self.aux_visual_objects = []

