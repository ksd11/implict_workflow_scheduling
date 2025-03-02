import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Tuple
from data.data_generate import DataGenerator
from .cluster import *
from pprint import pprint


class LayerEdgeDynamicEnv(gym.Env):
    def __init__(self, render_mode="human"):
        # N, L, C, Len
        # self.data = Data(5, 50, 20, 100)
        # self.data = Data(5, 500, 200, 100)
        generator = DataGenerator()
        generator.load("data/workload_data")
        pprint(generator.getSystemInfo())




        # self.N = data.N
        # self.L = data.L
        # N,L = data.N, data.L
        # obs_dim = N * (3*L+3) + 4 * N + L + 1
        # act_dim = N+1

        # self.observation_space = spaces.Box(
        #     low=0, high=math.inf, shape=(obs_dim,), dtype=np.float64)
        # self.action_space = spaces.Discrete(act_dim)
        # self.state = None

        # self.machines: list[Machine] = []
        # for idx, machine in enumerate(data.machines):
        #     self.machines.append(Machine(machine['cpu'], machine['storage'], machine['bandwidth'], data.layers, 2, idx))
        # # self.machines.append(Machine(data.cloud['cpu'], data.cloud['storage'], data.cloud['bandwidth'], data.layers))
        # self.layers = data.layers # layer_size的信息
        # self.cloud = Cloud(data.cloud['cpu'], data.cloud['storage'], data.cloud['bandwidth'], data.layers)

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
            reward = -finish_time/1000
        else:
            # to edge
            start_time, finish_time = self.machines[action].addTask(task, self.timestamp)
            reward = -finish_time/1000
        
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
    

LayerEdgeDynamicEnv()


