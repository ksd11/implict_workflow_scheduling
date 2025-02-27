import torch as th
import torch.nn as nn
from stable_baselines3 import DQN
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import Schedule
import gymnasium as gym
from typing import Optional,Type,List,Tuple

# 自定义 Policy 类
class CustomDQNPolicy(BasePolicy):
    def __init__(
        self,
        observation_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU, **kargs
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # 指定是否使用单独的特征提取器
            features_extractor=None, **kargs
        )
        
        # 定义网络结构
        if net_arch is None:
            net_arch = [64, 64]  # 默认隐藏层大小
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        
        # 创建 Q-Network
        self.q_net = self.make_q_net()
        self.q_net_target = self.make_q_net()
        self.q_net_target.load_state_dict(self.q_net.state_dict())
        
        # 优化器
        self.optimizer = th.optim.Adam(self.q_net.parameters(), lr=lr_schedule(1))

    def make_q_net(self) -> nn.Module:
        layers = []
        input_dim = self.observation_space.shape[0]
        for hidden_size in self.net_arch:
            layers.append(nn.Linear(input_dim, hidden_size))
            layers.append(self.activation_fn())
            input_dim = hidden_size
        layers.append(nn.Linear(input_dim, self.action_space.n))
        return nn.Sequential(*layers)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> Tuple[th.Tensor, th.Tensor]:
        q_values = self.q_net(obs)
        # 选择确定性或随机动作（DQN 通常选择 argmax）
        actions = q_values.argmax(dim=1, keepdim=True)
        return actions, q_values

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        with th.no_grad():
            actions, _ = self.forward(obs, deterministic)
        return actions

# 注册自定义 Policy（可选，仅用于自动识别）
# register_policy("CustomDQNPolicy", CustomDQNPolicy)

# # 创建使用自定义 Policy 的 DQN 模型
# model = DQN(
#     policy=CustomDQNPolicy,
#     env=env,
#     policy_kwargs={
#         "net_arch": [128, 128, 128],  # 自定义网络结构
#         "activation_fn": nn.Tanh,
#     },
#     verbose=1
# )