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
    def __init__(self, render_mode="human"
                 , need_log = False
                 , storage_type: Type[Storage] = PriorityStorage
                 , is_predeploy: bool = False
                 , predeploy_degree: int = 1
                 , prefix=None
                 , workload_data='data/workload_data'):
        generator = DataGenerator()
        generator.load(workload_data)
        # pprint(generator.getSystemInfo())
        self.data = generator
        self.totoal_server = len(self.data.nodes) # 总server的个数
        self.N = len(self.data.nodes)-1  # edge_Server的个数
        self.L = len(self.data.layers)   # layer的个数
        N,L = self.N, self.L
        # obs_dim = N * (3*L+3) + 4 * N + L + 1
        obs_dim = self.totoal_server * (6 + self.totoal_server) + 3
        act_dim = N+1
        self.Len = len(self.data.traces)
        self.need_log = need_log # 很耗时，统计时才打开

        self.is_predeploy = is_predeploy # 环境是否开启预部署
        self.predeploy_degree = predeploy_degree

        self.observation_space = spaces.Box(
            low=0, high=math.inf, shape=(obs_dim,), dtype=np.float64)
        self.action_space = spaces.Discrete(act_dim)
        self.state = None

        self.machines: list[Machine] = []
        for idx, node_info in enumerate(self.data.nodes):
            self.machines.append(Machine(idx, node_info, self.data, StorageCls=storage_type))
        
        # self.task_queue = TaskQueue()
        # self.__add_task_from_trace()

        # 预分配状态数组
        self._state_buffer = np.zeros(self.observation_space.shape[0], dtype=np.float64)

    # 从self.data.traces中添加任务
    def __add_task_from_trace(self):
        traces = self.data.traces

        # 将所有任务的起始任务添加到任务队列
        for idx, (timestamp, job_name, gen_pos) in enumerate(traces):
            G :nx.DiGraph = self.data.jobs[job_name]

            # is_dag = nx.is_directed_acyclic_graph(G)
            # assert is_dag, "bad dag, 因为dag有环"

            task_name = f"{job_name}_source"
            task_info = self.data.tasks_info[(job_name, task_name)]

            task = Task(job_name=job_name, task_name=task_name, arrival_time=timestamp, data=self.data, origin_pos=gen_pos, global_id=idx)

            self.task_queue.add_task(task)

    # - (finish_time - arrival_time)
    # 优化总的处理时间
    def reward_for_total_process_time(self, execution_info):
        return -(execution_info["finish_time"] - execution_info["arrival_time"])

    # 优化状态计算
    def __getState(self):
        self.__check_and_do_virtual_task()
        # 1. 预计算常用值
        task = self.task_queue.peek()
        if task is None:
            return self._state_buffer
        timestamp = task.get_arrival_time()
        # 2. 向量化机器状态计算
        machine_states = np.zeros((self.totoal_server, 6))
        machine_states[:, 0] = [m.cpu for m in self.machines]
        machine_states[:, 1] = [m.storage.remain() for m in self.machines]
        machine_states[:, 2] = [max(m.download_finish_time - timestamp, 0) for m in self.machines]
        machine_states[:, 3] = [m.getAddLayersSize(task.layer) * m.pull_dealy for m in self.machines]
        machine_states[:, 4] = [max(0, m.maxExistLayerDownloadTime(task.layer) - timestamp) for m in self.machines]

        # machine_states[:, 5] = [
        #     m.findEstByCore(m.get_ready_time(task)[0], task.cpu / m.cpu)[1] - timestamp for m in self.machines] # 机器最早开始执行的时间
        machine_states[:, 5] = [m.findEstByCore(timestamp, task.cpu / m.cpu)[1] - timestamp for m in self.machines]

        # 3. 批量更新状态缓冲区
        idx = 0
        self._state_buffer[idx:idx + self.totoal_server * 6] = machine_states.flatten()
        idx += self.totoal_server * 6
        self._state_buffer[idx:idx + self.totoal_server * self.totoal_server] = self.data.delay_matrix.flatten()
        idx += self.totoal_server * self.totoal_server
        self._state_buffer[idx:idx + 3] = [task.cpu, task.parent_pos, task.data_size]

        return self._state_buffer

    def __getState1(self):
        task = self.task_queue.peek()
        if task is None:
            return self._state_buffer  # 直接返回零数组
            
        timestamp = task.get_arrival_time()
        idx = 0

        # 机器信息
        for i, machine in enumerate(self.machines):
            add_layer_size = machine.getAddLayersSize(task)
            download_time = add_layer_size * machine.pull_dealy
            self._state_buffer[idx:idx+4] = [
                machine.cpu                              # cpu资源
                ,machine.storage.remain()                # 剩余空间的大小
                ,machine.download_finish_time - timestamp # 还需要下载的时间
                ,download_time                           # 增量下载的时间
            ]
            idx += 4
            self._state_buffer[idx:idx+self.totoal_server] = self.data.delay_matrix[i] # 和其他机器的通信延迟
            idx += self.totoal_server

        # 任务信息
        self._state_buffer[idx:idx+3] = [
            task.cpu                       # cpu 资源
            ,task.parent_pos               # 父节点位置
            ,task.data_size                # 传输数据大小
        ]

        return self._state_buffer
            

    def __getState_old(self):
        task = self.task_queue.peek()
        if task is None:
            return self._state_buffer  # 直接返回零数组
            
        timestamp = task.get_arrival_time()
        idx = 0
        
        # 批量更新机器状态
        for machine in self.machines[:-1]:
            # 使用切片赋值代替extend
            n_layers = len(self.data.layers)
            self._state_buffer[idx:idx + n_layers] = machine.has_layer
            idx += n_layers
            
            remaining_time = machine.getRemainingDownloadTime(timestamp)
            self._state_buffer[idx:idx + n_layers] = remaining_time
            idx += n_layers
            
            self._state_buffer[idx:idx + n_layers] = self.data.layers
            idx += n_layers
            
            self._state_buffer[idx:idx + 3] = [
                machine.cpu,
                machine.pull_dealy,
                machine.download_finish_time - timestamp
            ]
            idx += 3
        
        # 批量更新请求状态
        for machine in self.machines[:-1]:
            add_layer_size = machine.getAddLayersSize(task)
            download_time = add_layer_size * machine.pull_dealy
            waiting_time = max(timestamp, machine.download_finish_time) - timestamp
            compute_time = task.cpu / machine.cpu
            
            self._state_buffer[idx:idx + 4] = [
                add_layer_size,
                download_time,
                waiting_time,
                compute_time
            ]
            idx += 4
            
        # 任务状态
        self._state_buffer[idx:idx + len(task.has_layer)] = task.has_layer
        idx += len(task.has_layer)
        self._state_buffer[idx] = task.cpu
        
        return self._state_buffer


    def reset(self, seed=None, options={'trace_len':100}, return_info=None):
        trace_len = self.Len # 读取默认的
        if options is not None and 'trace_len' in options:
            trace_len = options['trace_len']

        self.data.traces = self.data.getNewTrace(seed=seed, trace_len = trace_len) # 初始化新的trace
        for machine in self.machines:
            machine.reset()
        self.clear_schedule_info()

        # add task
        self.task_queue = TaskQueue()
        self.__add_task_from_trace()
        
        return self.__getState(), {}
    
    # 移除任务，并将其后置任务添加到任务队列
    # 传入当前任务部署的位置，以及任务的完成时间
    def __next(self, pos, finish_time):
        task :Task = self.task_queue.get_task()

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
            data_size = G[task.task_name][chosen_succ].get('data_size', 0)
            
            # 添加选中的后继任务到队列
            new_task = Task(
                job_name=task.job_name,
                task_name=chosen_succ,
                arrival_time=finish_time, # 上一个任务结束之后才能开始
                data=self.data,
                parent_pos=pos,
                data_size=data_size,
                origin_pos=task.origin_pos,
                global_id=task.global_id
            )
            self.task_queue.add_task(new_task)
            return new_task

    # 执行虚拟任务，必须在origin_pos指定的位置执行
    def __check_and_do_virtual_task(self):
        task = self.task_queue.peek()
        while task is not None and (task.task_name.endswith("source") or task.task_name.endswith("sink")):
            self.step(task.origin_pos)
            task = self.task_queue.peek()

    def step(self, action, after_deploy_hook_func = None):
        reward = 0
        task = self.task_queue.peek()
        timestamp = task.get_arrival_time()
        if task == None:
            assert False, "Env is done!!"

        execution_info = self.machines[action].addTask(task)

        # reward = -execution_info["finish_time"]/1000

        # 优化总的处理时间
        reward = self.reward_for_total_process_time(execution_info)

        if self.need_log:
            self.record_schedule_info(
                global_id=task.global_id
                , task_id=task.get_task_id()
                , server_id=action, core_id=-1
                , **execution_info)
        
        # 在转移到下一个任务之前执行的操作，比如预拉取镜像层
        if after_deploy_hook_func != None:
            # 有自定义操作
            after_deploy_hook_func()
        elif self.is_predeploy:
            # 环境开启了预部署
            self.__predeploy(self.predeploy_degree)
        
        # 到下一个task
        new_task = self.__next(action, execution_info["finish_time"])

        # if self.is_predeploy and new_task != None:
        #     # 环境开启了预部署
        #     self.predeploy(timestamp, new_task.container_id)

        # 每次__getState()的时候就会排除所有虚拟任务
        return self.__getState(), reward, self.__idDone(), False, {"schedule_info": self.schedule_info}
    
    def record_schedule_info(self, global_id, task_id, server_id, core_id
                             , arrival_time, start_time, finish_time, wait_for_data, wait_for_image, wait_for_comp):
        self.schedule_info["tasks_execution_info"].append({
            "global_id": global_id, 
            "task_id": task_id,
            "server_id": server_id,
            "core_id": core_id,
            "arrival_time": arrival_time,
            "start_time": start_time,
            "finish_time": finish_time,
            "wait_for_data": wait_for_data,
            "wait_for_image": wait_for_image,
            "wait_for_comp": wait_for_comp
        })

        self.schedule_info["machines_info"] = []
        for i in range(self.totoal_server):
            self.schedule_info["machines_info"].append({
                "download_finish_time": self.machines[i].download_finish_time,
                "total_download_size": self.machines[i].total_download_size,
                # "data_transmission_time": self.machines[i].data_transmission_time
            })

    def clear_schedule_info(self):
        self.schedule_info = {"tasks_execution_info": [], "machines_info": []}

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
        return self.task_queue.is_empty()
    
    def get_sched_additional_info(self):
        task = self.task_queue.peek()
        if task == None:
            assert False, "Env is done!!"
        
        job_name = task.job_name
        G:nx.DiGraph = self.data.jobs[job_name]
        
        # 返回：任务达到时间，任务名，作业名，对应的dag图，任务信息表
        return task.arrival_time, task.task_name, job_name, G, self.data.tasks_info

    # 选择最早准备好的机器去预部署
    def predeploy(self, timestamp, container_id):
        # -1表示virtual container id
        if container_id == -1:
            return
        early_ready_time = math.inf # 最早准备好时间
        choose_mid = -1             # 选择部署的机器
        for mid, machine in enumerate(self.machines):
            ready_time = self.get_ready_time_of_container(mid, timestamp, container_id)
            if early_ready_time > ready_time:
                choose_mid = mid
                early_ready_time = ready_time
        self.predeploy_container(choose_mid, timestamp, container_id)

    # 预估的部署完成时间
    def get_ready_time_of_container(self, machine_id, timestamp, container_id):
        return self.machines[machine_id].get_ready_time_of_container(timestamp, container_id)

    # 预部署容器
    def predeploy_container(self, machine_id, timestamp, container_id):
        self.machines[machine_id].predeploy_container(timestamp, container_id)

    # 环境自己的预部署函数
    def __predeploy(self, predeploy_degree):
        timestamp, task_name, job_name, G, tasks_info = self.get_sched_additional_info()

        # 选择最有可能执行的函数预部署
        task_list:list[str] = []

        while len(task_list) < predeploy_degree:
            successors = list(G.successors(task_name))
            if len(successors) == 0:
                break
            probs = np.array([G[task_name][succ].get('probability', 1.0) 
                        for succ in successors])
            chosen_succ = successors[probs.argmax()]
            task_list.append(chosen_succ)
            task_name = chosen_succ

        # 从任务列表获取所有对应的容器
        container_list = [tasks_info[(job_name, tn)]["container_id"] for tn in task_list]
        # 预部署容器
        for container_id in container_list:
            self.predeploy(timestamp, container_id)

        
    


