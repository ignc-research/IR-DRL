import torch
from typing import Union, List

def directionalVectorsFromQuaternion(quaternion: Union[List, torch.Tensor], scale= 1) -> Union[List, torch.Tensor]:
    """
    Returns (scaled) up/forward/left vectors for world rotation frame quaternion.
    """
    x, y, z, w = quaternion
    up_vector = [
        scale*(2* (x*y - w*z)),
        scale*(1- 2* (x*x + z*z)),
        scale*(2* (y*z + w*x)),
    ]
    forward_vector = [
        scale*(2* (x*z + w*y)),
        scale*(2* (y*z - w*x)),
        scale*(1- 2* (x*x + y*y)),
    ]
    left_vector = [
        scale*(1- 2* (x*x + y*y)),
        scale*(2* (x*y + w*z)),
        scale*(2* (x*z - w*y)),
    ]

    if type(quaternion) is torch.Tensor:
        up_vector = torch.tensor(up_vector)
        forward_vector = torch.tensor(forward_vector)
        left_vector = torch.tensor(left_vector)

    return up_vector, forward_vector, left_vector

