

import numpy as np
import torch
import torch.nn.functional as F
from .trainer import Trainer,CfgType
import gymnasium as gym
from .util import rl_utils
from tqdm import tqdm
import os
    
class Qnet(torch.nn.Module):
    ''' 只有一层隐藏层的Q网络 '''
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qnet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
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

    def predict(self, state, deterministic=True):
        if deterministic:
            state = torch.tensor([state], dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
            return action, None
        else:
            return self.take_action(state), None


class CustomDQN(Trainer):
    def __init__(self, agent_cfg: CfgType, env_cfg: CfgType, train_cfg: CfgType):
        super(CustomDQN, self).__init__(agent_cfg, env_cfg, train_cfg)
        self.train_cfg["device"] = torch.device(self.train_cfg["device"])

        self.env = gym.make(**env_cfg)
        self.model = self._init_model()

    def _init_model(self):
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
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
                        next_state, reward, done, _, _ = self.env.step(action)
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

        # 保存模型到相应的目录
        self.model.save("./model/"+self.train_cfg["trainer_cls"]+"/"+ self.env_cfg["id"] +".pkl")

