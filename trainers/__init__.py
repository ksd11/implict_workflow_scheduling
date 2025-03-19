from .ppo import PPO
from .dqn import DQN
from .customdqn import CustomDQN
from .trainer import Trainer,CfgType
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor

def make_trainer(cfg, load=False):
    glob = globals()
    trainer_cls = cfg["trainer"]["trainer_cls"]
    assert trainer_cls in glob, f"'{trainer_cls}' is not a valid trainer."
    trainer:Trainer = glob[trainer_cls](
        agent_cfg=cfg["agent"], env_cfg=cfg["env"], train_cfg=cfg["trainer"]
    )
    if load:
        path = trainer.get_model_path()
        trainer.load(path, env=trainer.env, device=cfg["trainer"]["device"])
    return trainer

def model_path(model, env):
    return "./model/"+model+"/"+ env +".pkl"

def play_a_game(cfg):
    path = model_path(cfg['trainer']['trainer_cls'], cfg['env']['id'])
    trainer: Trainer = make_trainer(cfg, load=True)
    model = trainer.get_model()
    env = trainer.env

    # trainer.eval("Testing", n_eval_episodes=10, deterministic=True, render=False)
    # return None

    state,_ = env.reset()
    reward_sum = 0
    done = False

    while not done:
        action, _state = model.predict(state, deterministic=True)
        state, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        reward_sum += reward
        env.render()

    env.close()
    return reward_sum
