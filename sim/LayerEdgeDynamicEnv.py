import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Tuple
from data.data_generate import DataGenerator
from .cluster2 import *
from pprint import pprint
import networkx as nx


class LayerEdgeDynamicEnv(gym.Env):
    def __init__(self, render_mode="human"):
        generator = DataGenerator()
        generator.load("data/workload_data")
        # pprint(generator.getSystemInfo())
        self.data = generator
        self.N = len(self.data.nodes)-1
        self.L = len(self.data.layers)
        N,L = self.N, self.L
        obs_dim = N * (3*L+3) + 4 * N + L + 1
        act_dim = N+1

        self.observation_space = spaces.Box(
            low=0, high=math.inf, shape=(obs_dim,), dtype=np.float64)
        self.action_space = spaces.Discrete(act_dim)
        self.state = None

        self.machines: list[Machine] = []
        for idx, node_info in enumerate(self.nodes):
            self.machines.append(Machine(idx, node_info, self.data))
        
        self.task_queue = TaskQueue()
        self.__add_task_from_trace()

    # 从self.data.traces中添加任务
    def __add_task_from_trace(self):
        traces = self.data.traces

        # 将所有任务的起始任务添加到任务队列
        for timestamp, job_name in traces:
            G :nx.DiGraph = self.data.jobs[job_name]
            task_name = f"{job_name}_source"
            task_info = self.data.tasks_info[(job_name, task_name)]

            task = Task(job_name=job_name, task_name=task_name, task_info=task_info, arrival_time=timestamp, data=self.data)

            self.task_queue.add_task(task)


    def __getState(self):
        # 获取当前被调度的任务
        task:Task = self.task_queue.peek()
        timestamp = task.get_arrival_time()
        if task is None:
            return [0] * self.observation_space.shape[0]

        # for machine
        state = []
        for machine in self.machines:
            state.extend(machine.has_layer)
            state.extend(machine.getRemainingDownloadTime(timestamp))
            state.extend(self.data.layers)
            state.append(machine.cpu)
            state.append(machine.pull_dealy)
            state.append(machine.download_finish_time)
        
        # for current request
        for machine in self.machines:
            addLayerSize = machine.getAddLayersSize(task)
            state.append(addLayerSize) # 需要下载的大小
            state.append(addLayerSize * machine.pull_dealy) # 需要下载的时间
            state.append(max(timestamp, machine.download_finish_time)-timestamp) # waiting time
            state.append(task.cpu/machine.cpu) # 计算时间
        state.extend(task.has_layer) # 包含的层
        state.append(task.cpu) # request cpu resource

        return np.array(state)

    def reset(self, seed=None, options=None, return_info=None):
        self.data.traces = self.data.getNewTrace(seed) # 初始化新的trace
        for machine in self.machines:
            machine.reset()
        self.clear_schedule_info()
        return self.__getState(), {}
    
    # 移除任务，并将其后置任务添加到任务队列
    def __next(self):
        task :Task = self.task_queue.get_task()
        finish_time = task.get_finish_time()

        G :nx.DiGraph = self.data.jobs[task.job_name]
        
        # 获取所有直接后继节点
        successors = list(G.successors(task.task_name))
        
        if successors:
            # 获取所有边的概率
            probs = [G[task.task_name][succ].get('probability', 1.0) 
                    for succ in successors]
            
            # 确保概率和为1
            probs = np.array(probs) / sum(probs)
            
            # 按概率选择一个后继节点
            chosen_succ = np.random.choice(successors, p=probs)
            
            # 添加选中的后继任务到队列
            new_task = Task(
                job_name=task.job_name,
                task_name=chosen_succ,
                timestamp=finish_time, # 上一个任务结束之后才能开始
                data=self.data
            )
            self.task_queue.add_task(new_task)



    def step(self, action):
        reward = 0
        task = self.task_queue.peek()

        start_time, finish_time = self.machines[action].addTask(task)
        task.set_finish_time(finish_time)

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

    # 任务队列里面没任务代表完成了
    def __idDone(self) -> bool:
        return self.task_queue.peek() is None
    


