from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch as th

# 1. 自定义特征提取器
class FeatureExtractor(BaseFeaturesExtractor):
    # 输入observation，经过先行处理后输出features_dim维的特征
    def __init__(self, observation_space, features_dim=128):
        super().__init__(observation_space, features_dim)
        self.net = nn.Sequential(
            nn.Linear(observation_space.shape[0], 256),
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations):
        return self.net(observations)

class CustomAC(nn.Module):
    def __init__(
        self,
        features_dim: int,
        last_layer_dim_pi: int = 32,
        last_layer_dim_vf: int = 32,
    ):
        super().__init__()

        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        hidden_units_pi_1 = 64
        self.policy_net = nn.Sequential(
            nn.Linear(features_dim, hidden_units_pi_1),
            nn.ReLU(),
            nn.Linear(hidden_units_pi_1, last_layer_dim_pi),
            nn.ReLU(),
        )
        # Value network
        hidden_units_vf_1 = 64
        self.value_net = nn.Sequential(
            nn.Linear(features_dim, hidden_units_vf_1),
            nn.ReLU(),
            nn.Linear(hidden_units_vf_1, last_layer_dim_vf),
            nn.ReLU(),
        )

    def forward(self, features: th.Tensor):
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: th.Tensor) -> th.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: th.Tensor) -> th.Tensor:
        return self.value_net(features)


class CustomNetwork(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        features_dim = 100
        super().__init__(
            *args,
            **kwargs, 
            features_extractor_class=FeatureExtractor, 
            features_extractor_kwargs=dict(
                features_dim=features_dim,
            )
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = CustomAC(
            self.features_dim,
            last_layer_dim_pi=32,
            last_layer_dim_vf=32,
        )

