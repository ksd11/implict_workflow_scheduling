from abc import ABC, abstractmethod
from typing import Any
from stable_baselines3.common.evaluation import evaluate_policy
from sim.LayerEdgeEnv import LayerEdgeEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

CfgType = dict[str, Any]

from stable_baselines3.common.callbacks import EvalCallback, BaseCallback,StopTrainingOnNoModelImprovement

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

class Trainer(ABC):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        self.agent_cfg = agent_cfg
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg

    def train():
        pass
    
    # 根据env_cfg创建env环境
    def make_env(self, env_cfg):
        def get_env():
            env = LayerEdgeEnv()
            env = Monitor(env)  # 添加Monitor包装器
            return env
        # self.env = Monitor(gym.make(**env_cfg))
        # return get_env()
        return SubprocVecEnv([get_env for _ in range(8)], start_method='fork')
        # self.env = MyWrapper()
        # self.env = gym.make(**env_cfg)

    def pre_train(self):
        self.eval("Before train...")

    def post_train(self):
        self.eval("After train...")
        self.save(self.get_model_path())

    def save(self, path: str):
        # 保存模型到相应的目录
        print("Saving model to "+path)
        self.model.save(path)

    def load(self, path: str, env = None):
        print("loading model from "+path)
        # 有些模型会返回结果，有些不会
        res =  self.model.load(path, env=env)
        if res != None:
            self.model = res
        # self.model = self.model.__class__.load(path, env = self.env, print_system_info = True)
        return self
    
    def eval(self, msg, n_eval_episodes=5, deterministic=False, render=False):
        print(msg)
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=n_eval_episodes, deterministic=deterministic,render=render)
        print("mean_reward:",mean_reward,"std_reward:",std_reward)

    def get_model(self):
        return self.model
    
    # 训练过程中回调函数，记录感兴趣的信息
    def callback(self):
        # 创建 early stopping callback
        stop_train_callback = StopTrainingOnNoModelImprovement(
            max_no_improvement_evals=5,  # 5次评估没有提升则停止
            min_evals=5,               # 最少训练20次
            verbose=1
        )

        # early stopping callback
        eval_callback = EvalCallback(self.env,
                best_model_save_path=self.get_best_model_path(),
                log_path=self.get_best_model_path(), 
                eval_freq=5000,  # 每几步评估一次                    
                deterministic=True, 
                render=False,
                n_eval_episodes=5, # 每次评估运行5个episodes
                callback_after_eval=stop_train_callback,
                verbose=1 )
        tensorboard_callback = TensorboardCallback()

        return [eval_callback, tensorboard_callback]
    
    # 获取最终模型保存位置
    def get_model_path(self):
        return "./model/"+self.train_cfg["trainer_cls"]+"/"+ self.env_cfg["id"] +".pkl"
    
    # 获取最好结果保存路径
    def get_best_model_path(self):
        return "./model/"+self.train_cfg["trainer_cls"]+"/"+ self.env_cfg["id"]
    
    # 初始化模型
    def _init_model(self, model, train_cfg, params: list):
        model_kargs = {
            "env": self.env
        }
        for param in params:
            self._set(train_cfg, model_kargs, param)
        return model(**model_kargs)

    def _set(self, source, dest, key):
        if key in source:
            dest[key] = source[key]