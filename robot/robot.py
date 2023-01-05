from abc import ABC, abstractmethod

class Robot(ABC):

    # testing, change later
    def __init__(self):
        super().__init__()
        self.joints_ids = [1,2,3,4,5]
        self.name = "ur5_1"