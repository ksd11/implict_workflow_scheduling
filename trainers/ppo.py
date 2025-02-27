from stable_baselines3 import PPO as ST_PPO
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
# from leftenv import GoLeftEnv
from sim.LayerEdgeEnv import LayerEdgeEnv
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
import torch
from .trainer import Trainer,CfgType
from .network.layer_dependent_ppo import CustomNetwork
# from .network.custom_net import CustomNetwork
from .network.custom_cnn import CustomCNN
from sim.wrapper import MyWrapper
from .network.fm_net import FMNetwork
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


# class TensorboardCallback(BaseCallback):
#     def __init__(self, verbose=0):
#         super(TensorboardCallback, self).__init__(verbose)

#     def _on_step(self) -> bool:
#         # Log scalar value (here a random variable)
#         # value = np.random.random()
#         # self.logger.record('random_value', value)
#         return True

#     def _on_rollout_end(self) -> None:
#         # Log mean reward
#         mean_reward = np.mean(self.locals['rewards'])
#         self.logger.record('rollout/mean_reward', mean_reward)


# env_name = "CartPole-v0"
# env = gym.make(env_name)
# env = GoLeftEnv()
# env = LayerEdgeEnv()
# env = DummyVecEnv([lambda : env])
# env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

class PPO(Trainer):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        super(PPO, self).__init__(agent_cfg, env_cfg, train_cfg)

        def make_env():
            env = LayerEdgeEnv()
            env = Monitor(env)  # 添加Monitor包装器
            return env
        # self.env = Monitor(gym.make(**env_cfg))
        self.env = SubprocVecEnv([make_env for _ in range(8)], start_method='fork')
        # self.env = MyWrapper()
        # self.env = gym.make(**env_cfg)

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
    
        # eval_callback = EvalCallback(self.env, best_model_save_path='./model/',
                                    # log_path='./logs/', eval_freq=500,
                                    # deterministic=True, render=False)
        # tensorboard_callback = TensorboardCallback()

        # 开始训练
        # self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], progress_bar=["progress_bar"], callback=[eval_callback, tensorboard_callback])
        self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], progress_bar=["progress_bar"])
        
        self.post_train()


