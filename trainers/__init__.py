from .ppo import PPO
from .dqn import DQN
from .customdqn import CustomDQN
from .trainer import Trainer,CfgType
import gymnasium as gym

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
    model = make_trainer(cfg).load(path).get_model()

    env = gym.make(**cfg['env'])
    state,_ = env.reset()
    reward_sum = 0
    done = False

    while not done:
        action, _state = model.predict(state, deterministic=False)
        state, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        reward_sum += reward
        env.render()

    env.close()
    return reward_sum
