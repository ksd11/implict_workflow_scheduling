import torch as th
import torch.nn as nn
from stable_baselines3.dqn.policies import DQNPolicy
import torch.optim as optim

class CustomQNetwork(DQNPolicy):
    def __init__(self, observation_space, action_space, features_extractor=None, features_dim=256, lr=1e-3, **kargs):
        super().__init__(observation_space, action_space, features_extractor=features_extractor, features_dim=features_dim)

        print(kargs)

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
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

        self.q_net_target = self.q_net = nn.Sequential(
            nn.Linear(self.input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, action_space.n)
        )

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def forward(self, obs):
        # features = self.feature_extractor(obs)
        return self.q_net(obs)