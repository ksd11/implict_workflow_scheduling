

import numpy as np
import torch
import torch.nn.functional as F
from .trainer import Trainer,CfgType
import gymnasium as gym
from .util import rl_utils
from tqdm import tqdm
import os
from stable_baselines3.common.monitor import Monitor
from sim.wrapper import MyWrapper
    
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.state_dim = state_dim
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = x.reshape(-1, self.state_dim)
        x = F.relu(self.fc1(x))  # 隐藏层使用ReLU激活函数
        return self.fc2(x)
    
class CustomDQNModel:
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, learning_rate, gamma,
                 epsilon, target_update, device):
        self.action_dim = action_dim
        self.q_net = Qnet(state_dim, hidden_dim,
                          self.action_dim).to(device)  # Q网络
        # 目标网络
        self.target_q_net = Qnet(state_dim, hidden_dim,
                                 self.action_dim).to(device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(),
                                          lr=learning_rate)
        self.gamma = gamma  # 折扣因子
        self.epsilon = epsilon  # epsilon-贪婪策略
        self.target_update = target_update  # 目标网络更新频率
        self.count = 0  # 计数器,记录更新次数
        self.device = device

    def load(self, path):
        ''' 加载模型 '''
        self.q_net.load_state_dict(torch.load(path))
        self.target_q_net.load_state_dict(torch.load(path))

    def take_action(self, state):  # epsilon-贪婪策略采取动作
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.action_dim)
        else:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'],
                              dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions']).view(-1, 1).to(
            self.device)
        rewards = torch.tensor(transition_dict['rewards'],
                               dtype=torch.float).view(-1, 1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],
                                   dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'],
                             dtype=torch.float).view(-1, 1).to(self.device)

        q_values = self.q_net(states).gather(1, actions)  # Q值
        # 下个状态的最大Q值
        max_next_q_values = self.target_q_net(next_states).max(1)[0].view(
            -1, 1)
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones
                                                                )  # TD误差目标
        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  # 均方误差损失函数
        self.optimizer.zero_grad()  # PyTorch中默认梯度会累积,这里需要显式将梯度置为0
        dqn_loss.backward()  # 反向传播更新参数
        self.optimizer.step()

        if self.count % self.target_update == 0:
            self.target_q_net.load_state_dict(
                self.q_net.state_dict())  # 更新目标网络
        self.count += 1

    def save(self, path: str):
         # 获取目录路径
        directory = os.path.dirname(path)
        
        # 如果目录不存在则创建
        if directory and not os.path.exists(directory):
            os.makedirs(directory)
        torch.save(self.q_net.state_dict(), path)

    def predict(self, observation, state=None, episode_start=None, deterministic=True):
        observation = np.array(observation).reshape(1, -1)  # 确保是2D数组
            
        if deterministic:
            # 确定性预测
            with torch.no_grad():
                observation = torch.FloatTensor(observation).to(self.device)
                q_values = self.q_net(observation)
                action = q_values.argmax(dim=1).cpu().numpy()
        else:
            # epsilon-贪婪
            if np.random.random() < self.epsilon:
                action = np.array([np.random.randint(self.action_dim)])
            else:
                with torch.no_grad():
                    observation = torch.FloatTensor(observation).to(self.device)
                    q_values = self.q_net(observation)
                    action = q_values.argmax(dim=1).cpu().numpy()

        return action, state  # 返回形状为 (1,) 的numpy数组


class CustomDQN(Trainer):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        super(CustomDQN, self).__init__(agent_cfg, env_cfg, train_cfg)
        self.train_cfg["device"] = torch.device(self.train_cfg["device"])

        self.env = Monitor(gym.make(**env_cfg))
        # self.env = Monitor(MyWrapper())

        self.model = self._init_model()

    def _obs_space(self):
        # 1. Box空间 (连续)
        if isinstance(self.env.observation_space, gym.spaces.Box):
            if len(self.env.observation_space.shape) == 1:
                # 1D观察空间
                state_dim = self.env.observation_space.shape[0]
            else:
                # 多维观察空间 (如图像)
                state_dim = np.prod(self.env.observation_space.shape)
                
        # 2. Discrete空间 (离散)
        elif isinstance(self.env.observation_space, gym.spaces.Discrete):
            state_dim = 1
            
        # 3. Dict空间 (字典)
        elif isinstance(self.env.observation_space, gym.spaces.Dict):
            state_dim = sum(np.prod(space.shape) for space in self.env.observation_space.spaces.values())
        return state_dim
    
    def _action_space(self):
         # 动作空间
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            # Discrete(4) 表示有4个动作
            action_dim = self.env.action_space.n
        else:
            action_dim = self.env.action_space.shape[0]
        return action_dim

    def _init_model(self):
        # 3. 打印实际状态和动作示例
        # print(self.env.observation_space.sample())
        # print(self.env.action_space.sample())

        state_dim, action_dim = self._obs_space(), self._action_space()
        return CustomDQNModel(state_dim
                               , self.train_cfg["hidden_dim"]
                               , action_dim
                               , self.train_cfg["lr"]
                               , self.train_cfg["gamma"]
                               , self.train_cfg["epsilon"]
                               , self.train_cfg["target_update"]
                               , self.train_cfg["device"])

    def train(self):
        replay_buffer = rl_utils.ReplayBuffer(self.train_cfg["buffer_size"])
        return_list = []
        num_episodes = self.train_cfg["num_episodes"]

        for i in range(10):
            with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
                for i_episode in range(int(num_episodes / 10)):
                    episode_return = 0
                    state, _ = self.env.reset()
                    done = False
                    while not done:
                        action = self.model.take_action(state)
                        next_state, reward, termined, truncated, _ = self.env.step(action)
                        done = termined | truncated
                        replay_buffer.add(state, action, reward, next_state, done)
                        state = next_state
                        episode_return += reward
                        # 当buffer数据的数量超过一定值后,才进行Q网络训练
                        if replay_buffer.size() > self.train_cfg["minimal_size"]:
                            b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(self.train_cfg["batch_size"])
                            transition_dict = {
                                'states': b_s,
                                'actions': b_a,
                                'next_states': b_ns,
                                'rewards': b_r,
                                'dones': b_d
                            }
                            self.model.update(transition_dict)
                    return_list.append(episode_return)
                    if (i_episode + 1) % 10 == 0:
                        pbar.set_postfix({
                            'episode':
                            '%d' % (num_episodes / 10 * i + i_episode + 1),
                            'return':
                            '%.3f' % np.mean(return_list[-10:])
                        })
                    pbar.update(1)

        self.post_train()
        # 保存模型到相应的目录
        # self.model.save("./model/"+self.train_cfg["trainer_cls"]+"/"+ self.env_cfg["id"] +".pkl")

