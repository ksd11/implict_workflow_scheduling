from stable_baselines3 import PPO as ST_PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
# from leftenv import GoLeftEnv
from sim.LayerEdgeEnv import LayerEdgeEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
import torch
from .trainer import Trainer,CfgType


class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Log scalar value (here a random variable)
        # value = np.random.random()
        # self.logger.record('random_value', value)
        return True

    def _on_rollout_end(self) -> None:
        # Log mean reward
        mean_reward = np.mean(self.locals['rewards'])
        self.logger.record('rollout/mean_reward', mean_reward)


# env_name = "CartPole-v0"
# env = gym.make(env_name)
# env = GoLeftEnv()
# env = LayerEdgeEnv()
# env = DummyVecEnv([lambda : env])
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

class PPO(Trainer):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        super(PPO, self).__init__(agent_cfg, env_cfg, train_cfg)
        self.env = gym.make(**env_cfg)
        self.model = ST_PPO(
            policy=train_cfg["policy"],
            env=self.env,
            learning_rate=train_cfg["learning_rate"],
            n_steps=train_cfg["n_steps"], 
            batch_size=train_cfg["batch_size"],  #采样数据量
            n_epochs=train_cfg["n_epochs"],  #每次采样后训练的次数
            gamma=train_cfg["gamma"],
            verbose=train_cfg["verbose"],
            tensorboard_log=train_cfg["tensorboard_log"],
            device=train_cfg["device"])

    def train(self):
    
        eval_callback = EvalCallback(self.env, best_model_save_path='./model/',
                                    log_path='./logs/', eval_freq=500,
                                    deterministic=True, render=False)
        tensorboard_callback = TensorboardCallback()

        # 开始训练
        self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], progress_bar=["progress_bar"], callback=[eval_callback, tensorboard_callback])
        
        self.post_train()
