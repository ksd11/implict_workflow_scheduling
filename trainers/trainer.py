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

    def post_train(self):
        # 策略评估，可以看到倒立摆在平稳运行了
        mean_reward, std_reward = evaluate_policy(self.model, self.env, n_eval_episodes=10, render=False)
        #env.close()
        print("mean_reward:",mean_reward,"std_reward:",std_reward)

        # 保存模型到相应的目录
        self.model.save("./model/"+self.train_cfg["trainer_cls"]+"/"+ self.env_cfg["id"] +".pkl")

    def load(self, path: str):
        self.model.load(path)
        return self

    def get_model(self):
        return self.model