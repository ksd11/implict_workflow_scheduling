import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy
from .deepfm import Deepfm


class MlpLayer(nn.Module):
    def __init__(self, feature_dim, hidden_dims: list, dropout_rate=0.1):
        super(MlpLayer, self).__init__()
        
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

'''
policy network:
node feature走deepfm, 然后拼接起来作为新的node features
task feature直接走Mlp, 生成新的task features

[新node features + 新task features] --MLP--> 得到最终的输出


state_dim = 836, N = 5, L = 50
(L+L+L+1+1+1) * N + 4*N+L+1  = 765+71 = 836个特征
| node含有的layer信息L个 | layer的剩余下载时间L个 | layer大小L个 | 3 个和node的resource info相关的信息 |

前L个是sparse的特征

'''
class PolicyNetwork(nn.Module):
    def __init__(self, feature_dim, policy_net, dropout_rate=0.1, device="cuda"):
        super(PolicyNetwork, self).__init__()
        # state_dim = 836
        self.N = policy_net["N"]
        self.L = policy_net["L"]
        self.each_node_dim = 3*self.L+1+1+1
        self.node_feature_dim = self.each_node_dim * self.N 
        self.task_feature_dim = 4*self.N+self.L+1
        merge_feature_dim = 32 + self.N

        self.fm = self._build_deepfm(policy_net["node_layer"], device)

        self.task_extractor = MlpLayer(self.task_feature_dim, policy_net["task_layer"])
        self.merge_extractor = MlpLayer(merge_feature_dim, policy_net["merge_layer"])

    def _build_deepfm(self, dnn_hidden, device):
        sparse_features = []
        for i in range(self.L):
            sparse_features.append("C"+str(i))
        
        dense_features = []
        for i in range(self.L, self.each_node_dim):
            dense_features.append("I"+str(i))


        feat_sizes1={ feat:1 for feat in dense_features}
        feat_sizes2 = {feat: 2 for feat in sparse_features}
        feat_sizes={}
        feat_sizes.update(feat_sizes1)
        feat_sizes.update(feat_sizes2)
        return Deepfm(feat_sizes=feat_sizes, sparse_feature_columns=sparse_features, dense_feature_columns=dense_features, device=device, dnn_hidden_units=dnn_hidden)

    def forward(self,x):
        node_features = x[:,:self.node_feature_dim]
        task_features = x[:,self.node_feature_dim:]

        # 每个node特征经过deepfm
        node_deepfm_feature = []
        for i in range(self.N):
            each_node_feature = node_features[:, i*self.each_node_dim : (i+1)*self.each_node_dim]
            node_deepfm_feature.append(self.fm(each_node_feature))
        node_deepfm_feature = torch.cat(node_deepfm_feature, dim=1)

        # task特征经过MLP
        new_task_feature = self.task_extractor(task_features)

        # 聚合node特征和task特征
        features = torch.cat([node_deepfm_feature, new_task_feature], dim=1)
        
        # 聚合并经过MLP
        return self.merge_extractor(features)

class LayerDependentAC(nn.Module):
    def __init__(
        self,
        features_dim: int,
        net_arch: dict
    ):
        super().__init__()
        self.latent_dim_pi = net_arch["policy_net"]["merge_layer"][-1]
        self.latent_dim_vf = net_arch["value_net"][-1]

        # Policy network
        self.policy_net = PolicyNetwork(features_dim, net_arch["policy_net"])

        # Value network: 仅仅是简单的MLP
        self.value_net = MlpLayer(features_dim, net_arch["value_net"])

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


class CustomNetwork(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        # print(args)
        print(kwargs)
        self.my_net_arch = kwargs["net_arch"]
        super().__init__(
            *args,
            **kwargs
        )

    def _build_mlp_extractor(self) -> None:
        # print(self.features_dim) # 836
        self.mlp_extractor = LayerDependentAC(
            self.features_dim,
            net_arch=self.net_arch
        )


    