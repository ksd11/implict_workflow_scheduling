import torch as th
import torch.nn as nn
from stable_baselines3.dqn.policies import QNetwork

class CustomQNetwork(QNetwork):
    def __init__(self, observation_space, action_space, features_dim=256):
        super().__init__(observation_space, action_space)

        self.input_dim = observation_space.shape[0]
        
        # 特征提取器
        # self.feature_extractor = nn.Sequential(
        #     nn.Linear(observation_space.shape[0], 128),
        #     nn.ReLU(),
        #     nn.Linear(128, features_dim),
        #     nn.ReLU()
        # )
        
        # Q值网络
        self.q_net = nn.Sequential(
            nn.Linear(features_dim, 128),
            nn.ReLU(),
            nn.Linear(128, action_space.n)
        )

    def forward(self, obs):
        # features = self.feature_extractor(obs)
        return self.q_net(obs)