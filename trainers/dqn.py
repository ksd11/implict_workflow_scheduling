from stable_baselines3 import DQN as ST_DQN
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env import VecNormalize
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
from stable_baselines3.common.callbacks import EvalCallback, BaseCallback
import numpy as np
from .trainer import Trainer,CfgType
from stable_baselines3.common.monitor import Monitor
from .network.custom_dqn import CustomDQNPolicy
from .network.layer_dependent_dqn import LayerDependentExtractor

class TensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.episode_reward = 0

    def _on_step(self) -> bool:
         # 获取当前步骤的奖励
        reward = self.locals['rewards'][0]  # 对于非向量化环境是单个值
        self.episode_reward += reward
        
        # 检查是否episode结束
        done = self.locals['dones'][0]  # 获取结束信号
        if done:
            # 记录本episode的总奖励
            self.logger.record('episode/total_reward', self.episode_reward)
            # 重置累积奖励
            self.episode_reward = 0
        return True

    def _on_rollout_end(self) -> None:
        # 一个rollout不等于一次episode
        pass
        # Log mean reward
        # mean_reward = np.mean(self.locals['rewards'])
        # self.logger.record('rollout/mean_reward', mean_reward)
        
        # reward = np.sum(self.locals['rewards'])
        # self.logger.record('rollout/reward', reward)


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
        if "policy_kwargs" in train_cfg and "features_extractor_class" in train_cfg["policy_kwargs"]:
            extractor = train_cfg["policy_kwargs"]["features_extractor_class"]
            glob = globals()
            assert extractor in glob, f"'{extractor}' is not a valid extractor."
            train_cfg["policy_kwargs"]["features_extractor_class"] = glob[extractor]
        
        # train_cfg["policy"] = LayerDependentQNetwork
        self.model = self._init_model(model=ST_DQN, train_cfg=train_cfg, params=params)    

    def _set(self, source, dest, key):
        if key in source:
            dest[key] = source[key]

    def train(self):
        self.pre_train()

        tensorboard_callback = TensorboardCallback()
        # 开始训练
        self.model.learn(total_timesteps=self.train_cfg["total_timesteps"], 
                         progress_bar=self.train_cfg["progress_bar"], callback=[tensorboard_callback])
        self.post_train()
        

    


    