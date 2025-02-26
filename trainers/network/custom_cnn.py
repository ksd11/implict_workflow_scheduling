import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


#自定义特征抽取层
class CustomCNN(BaseFeaturesExtractor):

    def __init__(self, observation_space, hidden_dim):
        super().__init__(observation_space, hidden_dim)

        self.sequential = torch.nn.Sequential(

            #[b, 4, 1, 1] -> [b, h, 1, 1]
            torch.nn.Conv2d(in_channels=observation_space.shape[0],
                            out_channels=hidden_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),

            #[b, h, 1, 1] -> [b, h, 1, 1]
            torch.nn.Conv2d(hidden_dim,
                            hidden_dim,
                            kernel_size=1,
                            stride=1,
                            padding=0),
            torch.nn.ReLU(),

            #[b, h, 1, 1] -> [b, h]
            torch.nn.Flatten(),

            #[b, h] -> [b, h]
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )

    def forward(self, state):
        b = state.shape[0]
        state = state.reshape(b, -1, 1, 1)
        return self.sequential(state)