from typing import Callable, Dict, List, Optional, Tuple, Type, Union

from gym import spaces
import torch as th
from torch import nn

from stable_baselines3 import PPO
from stable_baselines3.common.policies import MultiInputActorCriticPolicy

class DropoutMultiInputActorCriticPolicy(MultiInputActorCriticPolicy):

    def __init__(self, *args, dropout_rate = 0.5, **kwargs):
        self.dropout_rate = dropout_rate
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = DropoutMlpExtractor(
            self.features_dim,
            self.device,
            self.dropout_rate,
        )

class DropoutMlpExtractor(nn.Module):
    """
    Custom network for policy and value function.
    It receives as input the features extracted by the features extractor.

    :param feature_dim: dimension of the features extracted with the features_extractor (e.g. features from a CNN)
    :param last_layer_dim_pi: (int) number of units for the last layer of the policy network
    :param last_layer_dim_vf: (int) number of units for the last layer of the value network
    """

    def __init__(
        self,
        feature_dim: int,
        device: Union[th.device, str] = "auto",
        dropout_rate: float = 0.5,
        last_layer_dim_pi: int = 64,
        last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Shared network
        self.shared_net = nn.Sequential(

        ).to(device)

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(feature_dim, last_layer_dim_pi), nn.Tanh(),
            nn.Linear(last_layer_dim_pi, last_layer_dim_pi), nn.Tanh(),
            
        ).to(device)
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.Tanh(),
            nn.Linear(feature_dim, last_layer_dim_vf), nn.Tanh(),
        ).to(device)

    def forward(self, features: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)