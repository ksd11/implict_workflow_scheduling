from abc import ABC, abstractmethod
from typing import Any
from stable_baselines3.common.evaluation import evaluate_policy
from sim.LayerEdgeEnv import LayerEdgeEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

CfgType = dict[str, Any]

class Trainer(ABC):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        self.agent_cfg = agent_cfg
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg

    def train():
        pass

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
        self.save("./model/"+self.train_cfg["trainer_cls"]+"/"+ self.env_cfg["id"] +".pkl")

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