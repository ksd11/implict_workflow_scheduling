import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


'''
网络只适用于layer aware环境
分为node特征和task特征

'''

class MlpFeatureExtra(nn.Module):
    def __init__(self, feature_dim, hidden_dims: list, dropout_rate=0.1):
        super(MlpFeatureExtra, self).__init__()
        
        hidden_dims = [feature_dim] + hidden_dims
        layers = []
        for i in range(len(hidden_dims)-1):
            layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if i != len(hidden_dims)-2:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(p=dropout_rate))  # 添加dropout层
        self.mlp = nn.Sequential(*layers)      

    def forward(self,x):
        return self.mlp(x)


class LayerDependentExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space
                 , features_dim: int = 128
                 , actions_dim: int = 5
                 , nodes_net_arch = [128, 64]
                 , tasks_net_arch = [128, 64]
                 , device = 'cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__(observation_space,  features_dim=features_dim)
        assert features_dim == nodes_net_arch[-1]+tasks_net_arch[-1], "dim error"
        self.device = device

        self.state_dim = observation_space.shape[0]
        self.action_dim = actions_dim
        self._set_node_and_task_feature_dim()
        # self._info()
        self.node_feature_extractor = MlpFeatureExtra(self.node_feature_dim, nodes_net_arch).to(self.device)
        self.task_feature_extractor = MlpFeatureExtra(self.task_feature_dim, tasks_net_arch).to(self.device)
        # self.merge_feature_extractor = MlpFeatureExtra(128, [64, features_dim])
        
    
    def _set_node_and_task_feature_dim(self):
        # state_dim = N(3L+3)+4N+L+1= (3L+7)N+L+1 = (3N+1)L+7N+1
        # action_dim = N+1
        N = self.action_dim - 1
        L = (self.state_dim-7*N-1) // (3*N+1)
        assert N*(3*L+3)+4*N+L+1 == self.state_dim, "dim error"
        self.node_feature_dim = N*(3*L+3)
        self.task_feature_dim = 4*N+L+1

    def _info(self):
        print("Using DQN...")
        print(f"state_dim: {self.state_dim}, action_dim: {self.action_dim}, node_feature_dim: {self.node_feature_dim}, task_feature_dim: {self.task_feature_dim}" )

    def forward(self, obs):
        obs = obs.to(self.device)
        self.node_input = obs[:,:self.node_feature_dim]
        self.task_input = obs[:,self.node_feature_dim:]
        node_output = self.node_feature_extractor(self.node_input)
        task_output = self.task_feature_extractor(self.task_input)
        
        # merge_input = torch.cat([node_output, task_output], dim=1)
        # return self.merge_feature_extractor(merge_input)
        return torch.cat([node_output, task_output], dim=1)
    