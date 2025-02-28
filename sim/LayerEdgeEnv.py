import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Tuple
from data.generate_data import Data

# np.random.seed(0)
# data = Data(5, 50, 20, 100)
# print(data.machines)
# print(data.cloud)
# print(data.layers)
# print(data.containers)
# print(data.trace)
# print("another: ")
# print(data.getAnotherTrace())

# print("=================================")

class Task:
    def __init__(self, task_id :int, container: dict, layer_size: list, arrival_time: float):
        self.task_id = task_id
        self.container = container
        self.cpu = container['cpu']
        self.layer = set(container['layer'])
        self.has_layer = []
        self.arrival_time = arrival_time
        for i in range(len(layer_size)):
            if(i in self.layer):
                self.has_layer.append(1)
            else:
                self.has_layer.append(0)

    def get_arrival_time(self):
        return self.arrival_time
    def get_task_id(self):
        return self.task_id

def ceil2(value):
    return math.ceil(value*100)/100

import portion as P
# 按至少0.01的粒度occupy，否则会有数值问题
class Core:
    def __init__(self, idx):
        self.idx = idx # 机器的第几个core
        self.interval = P.closedopen(0, P.inf)

    # 占据核[start, end)的资源
    def occupy(self, start, end):
        i = P.closedopen(start, end)
        # 假设已经被别人占领了，则无法占领
        if not self.interval.contains(i):
            print(self.interval)
            print(start, end)
            assert False, "occupy error"

        # 占领核[start, end)
        self.interval = self.interval - P.closedopen(start, end)
    
    def release(self, start, end):
        i = P.closedopen(start, end)
        if not (self.interval & i).empty:
            assert False, "release error" # 释放的是已经占据的
        self.interval = self.interval | i

    def is_occupy(self, start, end) -> bool:
        i = P.closedopen(start, end)
        return not self.interval.contains(i)
    
    def find_est(self, start, size) -> bool:
        for i in self.interval:
            real_start = max(i.lower, start)
            if i.upper >= real_start + size:
                return real_start
        assert False, "never be there"


    def __repr__(self):
        return self.interval.__str__()

    def __str__(self):
        return self.interval.__str__()

    def __iter__(self):
        return self.interval.__iter__()

class Machine:
    def __init__(self, cpu: float, storage: float, bandwidth: float, layer_size: list, core_number:int, idx: int):
        self.cpu = cpu
        self.storage = storage
        self.bandwidth = bandwidth
        self.L = len(layer_size)
        self.layer_size = layer_size
        self.core_number = core_number
        self.idx = idx
        self.reset()

    def reset(self):
        self.layers = {} # 记录对应layers的下载完成时间
        self.download_finish_time = 0
        # self.tasks = []
        # self.task_finish_time = 0
        self.total_download_size = 0
        self.has_layer = [0] * self.L
        self.cores = [Core(i) for i in range(self.core_number)]

    def getRemainingDownloadTime(self, timestamp: float):
        res = []
        for i in range(self.L):
            # 如果没有下载过或者已经下载完成
            if self.has_layer[i] == 0 or self.layers[i] <= timestamp:
                res.append(timestamp)
            else:
                res.append(self.layers[i])
        return res
    
    # 判断是否还能容纳此任务
    def isAccommodate(self, task: Task):
        if self.total_download_size + self.getAddLayersSize(task) > self.storage:
            return False
        # TODO. container number的限制如何做？
        return True
    
    def findEstByCore(self, start, size):
        res = math.inf
        core_id = -1
        for idx, core in enumerate(self.cores):
            est = core.find_est(start, size)
            if res > est:
                res = est
                core_id = idx
        return core_id, res
    
    def place(self, core_id, start, end):
        # print(f"edge[{self.idx}-{core_id}] occupy: {start}-{end}")
        self.cores[core_id].occupy(start, end)

                
    def addTask(self, task: Task, timestamp: float) -> Tuple[float, float]:
        # self.tasks.append(task)
        add_layers = self.getAddLayers(task)
        # 计算Layer下载完成时间
        ready_time = timestamp
        for layer in add_layers:
            self.layers[layer] = self.download_finish_time + self.layer_size[layer]/self.bandwidth
            # 记录信息
            self.download_finish_time = self.layers[layer]
            self.has_layer[layer] = 1
            self.total_download_size += self.layer_size[layer]
        if len(add_layers) > 0:
            ready_time = max(ready_time, self.download_finish_time)

        ready_time = ceil2(ready_time)
        
        # 计算Task完成时间
        execute_time = ceil2(task.cpu / self.cpu)
        core_id, est = self.findEstByCore(ready_time, execute_time)
        self.place(core_id, est, est+execute_time)
        # print(f"{timestamp:.2f}: executing task at [{est:.2f}-{est+execute_time:.2f}) in edge {self.idx}")
        # self.task_finish_time = max(self.download_finish_time, self.task_finish_time) + execute_time
        return est, est + execute_time

    def getAddLayers(self, task: Task):
        # 计算Layer下载完成时间
        layers = set(task.layer)
        add_layers = self.layers.keys() - layers
        return add_layers
    
    def getAddLayersSize(self, task: Task):
        return sum([self.layer_size[layer] for layer in self.getAddLayers(task)])

class Cloud(Machine):
    def __init__(self, cpu: float, storage: float, bandwidth: float, layer_size: list):
        super().__init__(cpu, storage, bandwidth, layer_size, 4, -1)

    def addTask(self, task: Task, timestamp: float) -> Tuple[float, float]:
        add_layers = self.getAddLayers(task)
        # 计算Layer下载完成时间
        ready_time = timestamp
        for layer in add_layers:
            self.layers[layer] = self.download_finish_time + self.layer_size[layer]/self.bandwidth
            # 记录信息
            self.download_finish_time = self.layers[layer]
            self.has_layer[layer] = 1
            self.total_download_size += self.layer_size[layer]
        if len(add_layers) > 0:
            ready_time = max(ready_time, self.download_finish_time)
        ready_time = ceil2(ready_time)

        # 计算Task完成时间
        execute_time = ceil2(task.cpu / self.cpu)
        est = ready_time
        # print(f"{timestamp:.2f}: executing task at [{est}-{est+execute_time}) in cloud")
        # self.task_finish_time = max(self.download_finish_time, self.task_finish_time) + execute_time
        return est, est + execute_time
    
    def isAccommodate(self, task: Task):
        return True

global_data = Data(5, 500, 200, 100)

class LayerEdgeEnv(gym.Env):
    def __init__(self, render_mode="human"):
        # N, L, C, Len
        # self.data = Data(5, 50, 20, 100)
        # self.data = Data(5, 500, 200, 100)
        self.data = global_data
        data = self.data
        self.N = data.N
        self.L = data.L
        N,L = data.N, data.L
        obs_dim = N * (3*L+3) + 4 * N + L + 1
        act_dim = N+1

        self.observation_space = spaces.Box(
            low=0, high=math.inf, shape=(obs_dim,), dtype=np.float64)
        self.action_space = spaces.Discrete(act_dim)
        self.state = None

        self.machines: list[Machine] = []
        for idx, machine in enumerate(data.machines):
            self.machines.append(Machine(machine['cpu'], machine['storage'], machine['bandwidth'], data.layers, 2, idx))
        # self.machines.append(Machine(data.cloud['cpu'], data.cloud['storage'], data.cloud['bandwidth'], data.layers))
        self.layers = data.layers # layer_size的信息
        self.cloud = Cloud(data.cloud['cpu'], data.cloud['storage'], data.cloud['bandwidth'], data.layers)

    def __getState(self):
        # 获取当前被调度的任务
        task = self.__getTask()
        if task is None:
            return [0] * self.observation_space.shape[0]

        # for machine
        state = []
        for machine in self.machines:
            state.extend(machine.has_layer)
            state.extend(machine.getRemainingDownloadTime(self.timestamp))
            state.extend(self.layers)
            state.append(machine.cpu)
            state.append(machine.bandwidth)
            state.append(machine.download_finish_time)
        
        # for current request
        for machine in self.machines:
            addLayerSize = machine.getAddLayersSize(task)
            state.append(addLayerSize) # 需要下载的大小
            state.append(addLayerSize / machine.bandwidth) # 需要下载的时间
            state.append(max(self.timestamp, machine.download_finish_time)-self.timestamp) # waiting time
            state.append(task.cpu/machine.cpu) # 计算时间
        state.extend(task.has_layer) # 包含的层
        state.append(task.cpu) # request cpu resource

        return np.array(state)

    def reset(self, seed=None, options=None, return_info=None):
        if seed is not None:
            np.random.seed(seed)
        self.timestamp = 0
        self.trace_idx = 0
        self.data.getAnotherTrace() # 初始化新的trace
        for machine in self.machines:
            machine.reset()
        self.cloud.reset()
        self.clear_schedule_info()
        return self.__getState(), {}
    
    def __getTask(self) -> Task:
        if self.__idDone():
            return None
        task_info = self.data.trace[self.trace_idx]
        task_id, arrival_time, container_id = task_info[0], task_info[1], int(task_info[2])
        container = self.data.containers[container_id]
        self.timestamp = arrival_time
        return Task(task_id=task_id, arrival_time=arrival_time, container=container, layer_size=self.layers )
    
    def __next(self):
        self.trace_idx += 1

    def step(self, action):
        reward = 0
        task = self.__getTask()
        if action == self.data.N:
            # to cloud
            start_time, finish_time = self.cloud.addTask(task, self.timestamp)
            reward = -finish_time
        else:
            # to edge
            start_time, finish_time = self.machines[action].addTask(task, self.timestamp)
            reward = -finish_time
        
        self.record_schedule_info(task_id=task.get_task_id()
                , server_id=action, core_id=-1
                , arrival_time=task.get_arrival_time()
                , start_time=start_time, finish_time=finish_time)
        
        # 到下一个task
        self.__next()
        return self.__getState(), reward, self.__idDone(), False, {"schedule_info": self.schedule_info}
    
    def record_schedule_info(self, task_id, server_id, core_id
                             , arrival_time, start_time, finish_time):
        self.schedule_info[task_id] = {
            "server_id": server_id,
            "core_id": core_id,
            "arrival_time": arrival_time,
            "start_time": start_time,
            "finish_time": finish_time
        }

    def clear_schedule_info(self):
        self.schedule_info = {}

    def get_schedule_info(self):
        return self.schedule_info
    
    # 判断动作是否合法，不合法需要重新sample
    def valid_action(self, action: int) -> bool:
        if(action == self.data.N):
            return True
        return self.machines[action].isAccommodate(self.__getTask())

    def render(self):
        # print(self.state)
        return 

    def __idDone(self) -> bool:
        return self.trace_idx == self.data.trace.shape[0]

if __name__ == '__main__':
    from stable_baselines3.common.env_checker import check_env

    env = LayerEdgeEnv()
    # It will check your custom environment and output additional warnings if needed
    print(check_env(env))

    env.reset(4)
    print(env.data.trace[:10])

    env.reset(8)
    print(env.data.trace[:10])

    env.reset(4)
    print(env.data.trace[:10])

