from stable_baselines3 import DQN as ST_DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
from .trainer import Trainer,CfgType

class DQN(Trainer):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        super(DQN, self).__init__(agent_cfg, env_cfg, train_cfg)
        self.env = gym.make(**self.env_cfg)
        self.model = ST_DQN(
            policy=train_cfg["policy"],      # MlpPolicy定义策略网络为MLP网络
            env=self.env, 
            learning_rate=train_cfg["learning_rate"],
            batch_size=train_cfg["batch_size"],
            buffer_size=train_cfg["buffer_size"],
            learning_starts=train_cfg["learning_starts"],
            target_update_interval=train_cfg["target_update_interval"],
            policy_kwargs={"net_arch" : train_cfg["policy_kwargs"]},     # 这里代表隐藏层为2层256个节点数的网络
            verbose=train_cfg["verbose"],                                   # verbose=1代表打印训练信息，如果是0为不打印，2为打印调试信息
            tensorboard_log=train_cfg["tensorboard_log"],  # 训练数据保存目录，可以用tensorboard查看
            device=train_cfg["device"]
        )

    def train(self):
        # 开始训练
        self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], 
                         progress_bar=self.train_cfg["progress_bar"])
        self.post_train()
        

    


    