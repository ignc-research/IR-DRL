import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

from .explanation import ExplainPPO
from sensor.lidar import LidarSensor

from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from zennit.rules import Stabilizer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim, preprocess_obs, maybe_transpose
from stable_baselines3.common.type_aliases import TensorDict
from typing import Callable

__all__ = [
    'VisualizeExplanations',
]

class VisualizeExplanations:
    
    def __init__(self, explainer : ExplainPPO, type_of_data = None):
        self.explainer = explainer
        self.type_of_data = type_of_data if type_of_data is not None else explainer.extractor_choice_bias
        self.open_figs = []

    def _color_fader(self, c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1=np.array(mplc.to_rgb(c1))
        c2=np.array(mplc.to_rgb(c2))
        return ((1-mix)*c1 + mix*c2)

    def _get_cmap(self, x, name= 'coolwarm'):
        return plt.get_cmap(name)(x)[:, :3]


    def _imshow_infer_type(self, data):
        data = data.squeeze()
        if len(data.shape) == 3:
            if min(data.shape) == 4: return 'rgbd'
            if min(data.shape) == 3: return 'rgb'
            if min(data.shape) == 1: return 'grayscale'
        raise NotImplementedError('data type inference is only available for camera data')
        if len(data.shape) == 1:
            if (data.shape[0] - 1) % 4 == 0 and data.shape[0] != 4: return 'lidar' # for now lidar seems to have 4 rays in parallel so fk it why not 

    def _update_imshow_rgbd(self, data, fig_rgbd, ims):
        im1, im2 = ims
        data : np.ndarray = data.squeeze().detach().numpy().transpose(1,2,0)
        rgb = data[:, :, :3].sum(axis=-1) + 0.5
        d = data[:, :, 3] + 0.5
        im1.set_data(rgb)
        im2.set_data(d)
        fig_rgbd.canvas.flush_events()

    def _start_imshow_rgbd(self, data):
        fig_rgbd, (ax_rgb, ax_d) = plt.subplots(1, 2)
        data : np.ndarray = data.squeeze().detach().numpy().transpose(1,2,0)
        rgb = data[:, :, :3].sum(axis= -1) + 0.5
        d = data[:, :, 3] + 0.5
        im1 = ax_rgb.imshow(rgb, vmin=0, vmax= 1, cmap='coolwarm')
        im2 = ax_d.imshow(d, vmin=0, vmax= 1, cmap='coolwarm')

        return fig_rgbd, (im1, im2)

    def _start_imshow_lidar(self, sensor, data,):
        self.sensor = sensor
        data = torch.clip(data + 0.5, 0, 1).squeeze().detach().numpy()
        if np.allclose(data, 0.5, atol = 0.1):
            self.sensor.render = False
        else:
            sensor.set_explanation_mode(True, self._get_cmap(data))
            self.sensor.render = True
        return None, None

    def _update_imshow_lidar(self, data,):
        data = torch.clip(data + 0.5, 0, 1).squeeze().detach().numpy()
        if np.allclose(data, 0.5, atol = 0.1):
            self.sensor.render = False
        else:
            self.sensor.set_explanation_mode(True, self._get_cmap(data))
            self.sensor.render = True

    def start_imshow(self, data, sensor_name : str = None):
        if self.type_of_data is None:
            self.type_of_data = self._imshow_infer_type(data)
        if self.type_of_data == 'rgbd':
            plt.ion()
            new_figs = self._start_imshow_rgbd(data,)
            self.open_figs.append(new_figs)
        if self.type_of_data == 'lidar':
            sensor_name = 'lidar' if sensor_name is None else sensor_name
            for sensor in self.explainer.env.sensors:
                if isinstance(sensor, LidarSensor) and sensor_name in sensor.output_name:
                    return self._start_imshow_lidar(sensor, data)
        
        return self.open_figs[-1]

    def update_imshow(self, data, fig = None, axs = None,):
        if self.type_of_data == 'rgbd':
            self._update_imshow_rgbd(data, fig, axs)
        if self.type_of_data == 'lidar':
            self._update_imshow_lidar(data)


    def start_imshow_from_obs(self, obs, extractor_name = None):
        data = self.explainer.explain_extractor(obs, extractor_name)
        self.type_of_data = self.type_of_data if extractor_name is None else extractor_name.split('_')[0]
        return self.start_imshow(data,)

    def update_imshow_from_obs(self, obs, fig = None, axs = None):
        data = self.explainer.explain_extractor(obs,)
        return self.update_imshow(data, fig, axs)

    def close_open_figs(self):
        for fig in self.open_figs:
            plt.close(fig[0])
