from stable_baselines3 import DQN as ST_DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
from .trainer import Trainer,CfgType
from stable_baselines3.common.monitor import Monitor

class DQN(Trainer):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        super(DQN, self).__init__(agent_cfg, env_cfg, train_cfg)
        self.env = Monitor(gym.make(**self.env_cfg))
        self.model = self._init_model(train_cfg)

    def _init_model(self, train_cfg):
        model_kargs = {
            "env": self.env
        }

        params = [
            "policy","learning_rate", "batch_size", "buffer_size", "learning_starts","target_update_interval",
            "policy_kwargs","verbose","tensorboard_log","tau","gamma","device"
        ]

        for param in params:
            self._set(train_cfg, model_kargs, param)

        return ST_DQN(**model_kargs)

    def _set(self, source, dest, key):
        if key in source:
            dest[key] = source[key]

    def train(self):
        # 开始训练
        self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], 
                         progress_bar=self.train_cfg["progress_bar"])
        self.post_train()
        

    


    