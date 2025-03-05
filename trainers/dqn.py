from stable_baselines3 import DQN as ST_DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import numpy as np
from .trainer import Trainer,CfgType
from stable_baselines3.common.monitor import Monitor
from .network.custom_dqn import CustomDQNPolicy
from stable_baselines3.common.utils import get_linear_fn

class DQN(Trainer):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        super(DQN, self).__init__(agent_cfg, env_cfg, train_cfg)
        self.env = self.make_env(env_cfg)
        params = [
            # https://stable-baselines3.readthedocs.io/en/master/modules/dqn.html
            "policy" # MlpPolicy, CnnPolicy, 自定义
            ,"learning_rate"
            , "batch_size"
            , "buffer_size"
            , "gamma"
            , "tau"            #  the soft update coefficient 
            , "learning_starts" # how many steps of the model to collect transitions for before learning starts
            , "target_update_interval"
            , "policy_kwargs" # 对应不同的policy可以指定不同的网络结构
            , "verbose"
            , "tensorboard_log"
            , "device"
            , "exploration_initial_eps"
            , "exploration_final_eps"
        ]
        
        # initial_lr = 1e-3
        # final_lr = 1e-5
        # lr_schedule = get_linear_fn(
        #     initial_lr,
        #     final_lr,
        #     1
        # )
        # train_cfg["learning_rate"] = lr_schedule
        self.model = self._init_model(model=ST_DQN, train_cfg=train_cfg, params=params)    

    def _set(self, source, dest, key):
        if key in source:
            dest[key] = source[key]

    def train(self):
        self.pre_train()

        # # 开始训练
        self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], 
                         progress_bar=self.train_cfg["progress_bar"], callback=self.callback())
        self.post_train()
        

    


    