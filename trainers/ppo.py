from stable_baselines3 import PPO as ST_PPO
import gymnasium as gym
# from leftenv import GoLeftEnv
import numpy as np
import torch
from .trainer import Trainer,CfgType
from .network.layer_dependent_ppo import CustomNetwork
# from .network.custom_net import CustomNetwork
from .network.custom_cnn import CustomCNN
from sim.wrapper import MyWrapper
from .network.fm_net import FMNetwork
from stable_baselines3.common.utils import get_linear_fn


# env_name = "CartPole-v0"
# env = gym.make(env_name)
# env = GoLeftEnv()
# env = LayerEdgeEnv()
# env = DummyVecEnv([lambda : env])
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

class PPO(Trainer):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        super(PPO, self).__init__(agent_cfg, env_cfg, train_cfg)

        self.env = self.make_env(env_cfg)

        params = [
            "policy"
            , "learning_rate"
            , "n_steps"
            , "batch_size"
            , "n_epochs"
            , "policy_kwargs"
            , "gamma"
            , "verbose"
            , "tensorboard_log"
            , "device"
            , "clip_range"
            , "gae_lambda"
            , "ent_coef"
            , "vf_coef"
            , "max_grad_norm"
            , "target_kl"
        ]
        # train_cfg["policy"] = CustomNetwork
        # 1. 创建学习率衰减函数
        initial_lr = train_cfg.get("learning_rate", 3e-4)
        end_lr = initial_lr * 0.1  # 最终学习率为初始值的10%
        
        # 线性衰减
        self.lr_schedule = get_linear_fn(
            initial_lr,  # 初始学习率
            end_lr,     # 最终学习率
            1           # 总进度为1
        )
        # 2. 更新训练配置
        train_cfg["learning_rate"] = self.lr_schedule

        self.model = self._init_model(model=ST_PPO, train_cfg=train_cfg, params=params)

    def train(self):
        self.pre_train()

        # 开始训练
        self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], progress_bar=["progress_bar"], callback=self.callback())
        
        self.post_train()


