
from stable_baselines3.common.policies import ActorCriticPolicy
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch

class FM(nn.Module):
    def __init__(self, feature_size, k=10):
        """
        FM模型
        Args:
            feature_size: 特征数量
            k: 隐向量维度
        """
        super(FM, self).__init__()
        self.w0 = nn.Parameter(torch.zeros([1]))
        self.w1 = nn.Parameter(torch.randn([feature_size, 1]) * 0.01)
        self.v = nn.Parameter(torch.randn([feature_size, k]) * 0.01)

    def forward(self, x):
        """
        前向传播
        Args:
            x: 输入特征 [batch_size, feature_size]
        """
        # 一阶特征
        first_order = torch.matmul(x, self.w1)
        
        # 二阶特征交叉
        square_of_sum = torch.pow(torch.matmul(x, self.v), 2)
        sum_of_square = torch.matmul(torch.pow(x, 2), torch.pow(self.v, 2))
        second_order = 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)

        # 输出
        out = self.w0 + first_order + second_order
        out = torch.sigmoid(out)
        return out
    

class FMAC(nn.Module):
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
            FM(features_dim),
            nn.Linear(1, last_layer_dim_pi),
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

    def forward(self, features: torch.Tensor):
        """
        :return: (th.Tensor, th.Tensor) latent_policy, latent_value of the specified network.
            If all layers are shared, then ``latent_policy == latent_value``
        """
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        return self.policy_net(features)

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        return self.value_net(features)


class FMNetwork(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        features_dim = 100
        super().__init__(
            *args,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = FMAC(
            self.features_dim,
            last_layer_dim_pi=32,
            last_layer_dim_vf=32,
        )

