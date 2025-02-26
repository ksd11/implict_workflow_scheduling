from .ppo import PPO
from .dqn import DQN
from .customdqn import CustomDQN
from .trainer import Trainer,CfgType
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def make_trainer(cfg):
    glob = globals()
    trainer_cls = cfg["trainer"]["trainer_cls"]
    assert trainer_cls in glob, f"'{trainer_cls}' is not a valid trainer."
    return glob[trainer_cls](
        agent_cfg=cfg["agent"], env_cfg=cfg["env"], train_cfg=cfg["trainer"]
    )

def model_path(model, env):
    return "./model/"+model+"/"+ env +".pkl"

def play_a_game(cfg):
    path = model_path(cfg['trainer']['trainer_cls'], cfg['env']['id'])
    trainer: Trainer = make_trainer(cfg)
    model = trainer.load(path).get_model()
    env = trainer.env

    trainer.eval("Testing", n_eval_episodes=10, deterministic=True)

    return None

    # state,_ = env.reset()
    # reward_sum = 0
    # done = False

    # while not done:
    #     action, _state = model.predict(state, deterministic=True)
    #     state, reward, terminated, truncated, info = env.step(int(action))
    #     done = terminated or truncated
    #     reward_sum += reward
    #     env.render()

    # env.close()
    # return reward_sum
