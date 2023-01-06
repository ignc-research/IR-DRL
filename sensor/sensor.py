from abc import ABC, abstractmethod
from time import time

class Sensor(ABC):
    """
    Abstract Base Class for a sensor. The methods signed with abstractmethod have to be implemented in subclasses.
    See the joint or position sensor for examples.
    """

    def __init__(self, normalize: bool, add_to_observation_space: bool=True):
        
        super().__init__()
        
        # determines whether the output of the sensor is normalized to be between -1 and 1 (or alternatively between 0 and 1, if that makes more sense for a particular type of sensor)
        # note: class variables as well as the logging output should still be unnormalized, only the output of get_data() should be changed by this
        self.normalize = normalize
        
        # determines wether this sensor will create a field for the observation space, default is yes
        # useful to set to false if you need the sensor data for something but don't want the model to have access to it
        self.add_to_observation_space = add_to_observation_space

        # sets the epoch, use this to determine relative time, useful for e.g. velocities
        self.time = 0
        self.epoch = time()

        # at the end of init the sensor should also update itself
        # self.update()  # add this in your subclass at the end of __init__ without the comment

    @abstractmethod
    def update(self) -> dict:
        """
        Updates sensor data by performing underlying data collection.
        This also includes any operations necessary to adapt the sensor's state to the changed environment, e.g. in case of moving sensors.
        Returns the current data.
        """
        pass

    @abstractmethod
    def get_data(self) -> dict:
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

    @abstractmethod
    def get_data_for_logging(self) -> dict:
        """
        This method can be used to return data in a format that is more useful for logging.
        Besides the primary data returned by get_data(), this can also include secondary data (e.g. the position sensor does not return absolute position in get_data, so this is a way to make that data available).
        Output should be a dict with parameter name as key and the value as value.
        If there is no use case for this with a specific sensor implementation, return an empty dict instead.
        You can also add custom methods for returning specific pieces of data other than this method, however only this method will be used for automatic logging.
        """
        pass

