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
        ]
        train_cfg["policy"] = CustomNetwork
        self.model = self._init_model(model=ST_PPO, train_cfg=train_cfg, params=params)

    def train(self):
        self.pre_train()

        # 开始训练
        self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], progress_bar=["progress_bar"], callback=self.callback())
        
        self.post_train()


