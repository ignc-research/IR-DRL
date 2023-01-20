import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

from .explanation import ExplainPPO

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
    
    def __init__(self, explainer : ExplainPPO):
        self.explainer = explainer
        self.open_figs = []

    def _color_fader(c1,c2,mix=0): #fade (linear interpolate) from color c1 (at mix=0) to c2 (mix=1)
        c1=np.array(mplc.to_rgb(c1))
        c2=np.array(mplc.to_rgb(c2))
        return mplc.to_hex(((1-mix)*c1 + mix*c2))


    def _imshow_infer_type(self, data):
        data = data.squeeze()
        if len(data.shape) == 3:
            if min(data.shape) == 4: return 'rgbd'
            if min(data.shape) == 3: return 'rgb'
            if min(data.shape) == 1: return 'grayscale'
        if len(data.shape) == 1:
            if (data.shape[0] - 1) % 4 == 0 and data.shape[0] != 4: return 'lidar' # for now lidar seems to have 4 rays in parallel so fk it why not 

    def _update_imshow_rgbd(self, data, fig_rgbd, ims):
        im1, im2 = ims
        data : np.ndarray = data.squeeze().numpy().transpose(1,2,0)
        rgb = data[:, :, :3].sum(axis=-1) + 0.5
        d = data[:, :, 3] + 0.5
        im1.set_data(rgb)
        im2.set_data(d)
        fig_rgbd.canvas.flush_events()

    def _start_imshow_rgbd(self, data):
        fig_rgbd, (ax_rgb, ax_d) = plt.subplots(1, 2)
        data : np.ndarray = data.squeeze().numpy().transpose(1,2,0)
        rgb = data[:, :, :3].sum(axis= -1) + 0.5
        d = data[:, :, 3] + 0.5
        im1 = ax_rgb.imshow(rgb, vmin=0, vmax= 1, cmap='coolwarm')
        im2 = ax_d.imshow(d, vmin=0, vmax= 1, cmap='coolwarm')

        return fig_rgbd, (im1, im2)



    def start_imshow(self, data, type_of_data= None, ):
        plt.ion()
        if type_of_data is None:
            type_of_data = self._imshow_infer_type(data)
        if type_of_data == 'rgbd':
            self.open_figs.append(self._start_imshow_rgbd(data,))
        if type_of_data == 'lidar':
            self.explainer
        
        return self.open_figs[-1]

    def update_imshow(self, data, fig, axs, type_of_data = None, ):
        if type_of_data is None:
            type_of_data = self._imshow_infer_type(data)
        if type_of_data == 'rgbd':
            self._update_imshow_rgbd(data, fig, axs)

    def start_imshow_from_obs(self, obs, extractor_name = None):
        data = self.explainer.explain_extractor(obs, extractor_name)
        return self.start_imshow(data,)

    def update_imshow_from_obs(self, obs, fig, axs):
        data = self.explainer.explain_extractor(obs,)
        return self.update_imshow(data, fig, axs)

    def close_open_figs(self):
        for fig in self.open_figs:
            plt.close(fig[0])
