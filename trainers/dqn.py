from stable_baselines3 import DQN as ST_DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
from .trainer import Trainer,CfgType
from stable_baselines3.common.monitor import Monitor
from .network.custom_dqn import CustomQNetwork
from .network.layer_dependent_dqn import LayerDependentQNetwork

class DQN(Trainer):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        super(DQN, self).__init__(agent_cfg, env_cfg, train_cfg)
        self.env = Monitor(gym.make(**self.env_cfg))
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
        ]
        # train_cfg["policy"] = LayerDependentQNetwork
        train_cfg["policy"] = CustomQNetwork
        self.model = self._init_model(model=ST_DQN, train_cfg=train_cfg, params=params)    

    def _set(self, source, dest, key):
        if key in source:
            dest[key] = source[key]

    def train(self):
        self.pre_train()
        # 开始训练
        self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], 
                         progress_bar=self.train_cfg["progress_bar"])
        self.post_train()
        

    


    