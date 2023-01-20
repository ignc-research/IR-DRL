import torch
import numpy as np
import gym
import matplotlib.pyplot as plt
import matplotlib.colors as mplc

from zennit.composites import EpsilonGammaBox
from zennit.canonizers import SequentialMergeBatchNorm
from zennit.attribution import Gradient
from zennit.rules import Stabilizer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, NatureCNN
from stable_baselines3.common.preprocessing import is_image_space, get_flattened_obs_dim, preprocess_obs, maybe_transpose
from stable_baselines3.common.type_aliases import TensorDict
from typing import Callable

__all__ = [
    'ExplainPPO',
]

class ExplainPPO:

    def __init__(self, env, model, current_extractor_name = None, composite = None, normalize_images = False, extractor_bias= 'camera'):
        self.policy = model.policy
        self.obs_space = env.observation_space
        self.normalize_images = normalize_images
        self.extractors = self.policy.features_extractor.extractors
        self.current_extractor = None
        self.extractor_choice_bias = extractor_bias
        self.current_extractor = self.setup_extractor(current_extractor_name)
        obs = self._preprocess_obs()
        self.feature_indices = self._get_feature_indices()
        self.zero_obs = {key : torch.zeros_like(val) for key, val in obs.items()}
        self.zero_features = self.policy.features_extractor(self.zero_obs) # check whether its working to prevent unexpected runtime errors 

        shared = self.policy.mlp_extractor.shared_net # deprecated but there for compatibility
        if len(shared) > 0:
            self.value_net = torch.nn.Sequential(shared, self.policy.mlp_extractor.value_net, self.policy.value_net)
            self.action_net = torch.nn.Sequential(shared, self.policy.mlp_extractor.policy_net, self.policy.action_net)
        else:
            self.value_net = torch.nn.Sequential(self.policy.mlp_extractor.value_net, self.policy.value_net)
            self.action_net = torch.nn.Sequential(self.policy.mlp_extractor.policy_net, self.policy.action_net)

        if composite is None:
            self.composite = EpsilonGammaBox(
                low=-3.,
                high=3.,
                epsilon=1e-4,
                gamma=2.,
                stabilizer=Stabilizer(epsilon=1e-5, clip=True),
                zero_params='bias',
            )
        else: self.composite = composite

    def _preprocess_obs(self, obs : np.ndarray = None):
        d = {}
        if obs is None:
            obs = {key: value.sample() for key, value in self.obs_space.items()}
        for key, val in obs.items():
            if len(val.shape) == 3 and val.shape[0] > val.shape[2]:
                val = val.transpose(2,0,1)
            obs = torch.tensor(val).unsqueeze(0)
            d[key] = obs
        obs = preprocess_obs(d, self.obs_space, self.normalize_images)
        keys = obs.keys()
        vals = []
        for v in obs.values():
            val = v.clone()
            val.requires_grad = True
            vals.append(val)

        return {key:val for key,val in zip(keys, vals)}

    def _get_feature_indices(self):
        obs = self._preprocess_obs()
        out_size = self.policy.features_extractor(obs).shape[1]
        feature_indices = {}
        mark = 0
        for key, extractor in self.extractors.items():
            size = extractor(obs[key]).shape[1]
            feature_indices[key] = torch.arange(size) + mark
            mark += size

        return feature_indices

    def _automatically_choose_extractor(self):
        if self.current_extractor is None:
            for key in self.obs_space.keys():
                if self.extractor_choice_bias in key:
                    extractor_name = key
                    break
            print(f'Extractor: "{extractor_name}" was automatically selected.')
            self.current_extractor = self.extractors[extractor_name]
            self.current_extractor_name = extractor_name
        else:
            pass

    def setup_extractor(self, extractor_name = None):
        if extractor_name is None:
            self._automatically_choose_extractor()
        elif extractor_name not in self.extractors.keys():
            print(f'Extractor "{extractor_name}" is not recognized.')
            self._automatically_choose_extractor()
        else:
            self.current_extractor = self.extractors[extractor_name]
            self.current_extractor_name = extractor_name

    def _explain_extractor_inner(self, obs, attribution):
        with self.composite.context(self.value_net) as modified_model:
            obs = self._preprocess_obs(obs)
            output = modified_model(obs)
            if grad_outputs is None:
                grad_outputs = torch.ones_like(output)
            # compute the attribution via the gradient
            attribution, = torch.autograd.grad(
                output, obs, grad_outputs=grad_outputs
            )
        return attribution

    def explain_extractor(self, obs, extractor_name= None, value_or_action= 'value', grad_outputs= None):
        self.setup_extractor(extractor_name)
        if value_or_action == 'value':
            attribution = self.explain_value_net(obs, grad_outputs, create_graph= True)
        elif value_or_action == 'action':
            attribution = self.explain_action_net(obs, grad_outputs, create_graph= True)
        elif value_or_action == 'debug':
            attribution = self._explain_extractor_inner(obs, grad_outputs)
        else:
            raise ValueError(f'value_or_action has to be either "value" or "action". crazy right?')
        obs = self._preprocess_obs(obs)
        with self.composite.context(self.current_extractor) as modified_model:
            obs = obs[self.current_extractor_name]
            _divide = modified_model(obs)
            output = attribution[:,self.feature_indices[self.current_extractor_name]]
            attribution, = torch.autograd.grad(
                _divide, obs, grad_outputs=output, allow_unused= True,
            )
        return attribution

    def _emulate_features_extractor(self, obs, extractor_name= None):
        self.setup_extractor(extractor_name)
        emulated_obs = {}
        for key, val in self.zero_obs.items():
            if key != self.current_extractor_name:
                t = val
            elif key == self.current_extractor_name:
                t = obs[key]
            t.requires_grad = True
            t.grad = None
            emulated_obs[key] = t

        return emulated_obs,

    def explain_extractor_old(self, obs, grad_outputs= None):
        
        obs = self._preprocess_obs(obs)
        with self.composite.context(self.policy) as modified_model:
            output = modified_model(obs)
            if grad_outputs is None:
                grad_outputs = torch.ones_like(output)
            # compute the attribution via the gradient
            attribution, = torch.autograd.grad(
                output, obs, grad_outputs=grad_outputs
            )
        return attribution

    def explain_value_net(self, obs, grad_outputs= None, create_graph= False):
        """
        Explains the value_net until the output of the features_extractor
        """
        obs = self._preprocess_obs(obs)
        features = self.policy.features_extractor(obs)
        with self.composite.context(self.value_net) as modified_model:
            output = modified_model(features)
            if grad_outputs is None:
                grad_outputs = torch.ones_like(output)
            # compute the attribution via the gradient
            attribution, = torch.autograd.grad(
                output, features, grad_outputs=grad_outputs, create_graph=create_graph,
            )
        return attribution
        
    def explain_action_net(self, obs, grad_outputs= None, create_graph= False):
        """
        Explains the action_net until the output of the features_extractor
        """
        obs = self._preprocess_obs(obs)
        features = self.policy.features_extractor(obs)
        with self.composite.context(self.action_net) as modified_model:
            output = modified_model(features)
            if grad_outputs is None:
                grad_outputs = torch.ones_like(output)
            # compute the attribution via the gradient
            attribution, = torch.autograd.grad(
                output, features, grad_outputs=grad_outputs, create_graph=create_graph,
            )
        return attribution