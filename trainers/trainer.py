from abc import ABC, abstractmethod
from typing import Any
from stable_baselines3.common.evaluation import evaluate_policy

CfgType = dict[str, Any]

class Trainer(ABC):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        self.agent_cfg = agent_cfg
        self.env_cfg = env_cfg
        self.train_cfg = train_cfg

    def train():
        pass

    def pre_train(self):
        self.eval("Before train...")

    def post_train(self):
        self.eval("After train...")
        self.save("./model/"+self.train_cfg["trainer_cls"]+"/"+ self.env_cfg["id"] +".pkl")

    def save(self, path: str):
        # 保存模型到相应的目录
        print("Saving model to "+path)
        self.model.save(path)

    def load(self, path: str):
        print("loading model from "+path)
        self.model.load(path)
        return self
    
    def eval(self, msg, n_eval_episodes=30, deterministic=False, render=False):
        print(msg)
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=n_eval_episodes, deterministic=deterministic,render=render)
        print("mean_reward:",mean_reward,"std_reward:",std_reward)

    def get_model(self):
        return self.model