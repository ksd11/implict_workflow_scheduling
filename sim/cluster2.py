import math
from typing import Tuple
from collections import OrderedDict
from data.data_generate import DataGenerator

import heapq
from typing import Any, Optional

class Task:
    '''
        job_name和task_name唯一确定一个任务
        arrival_time: 任务到达的时间（父任务执行完成并选择该子任务执行）
        data: 全局信息，方便获取数据
        parent_pos和data_size确定了数据传输过来的时间：
            data_ready_time = arrival_time + dealy[parent_pos][cur] * data_size
            任务不能在数据准备好之前执行
    '''
    def __init__(self, job_name: str, task_name :str, arrival_time: float, data: DataGenerator, parent_pos: int = 0, data_size: float = 0):
        task_info = data.tasks_info[(job_name,task_name)]
        self.job_name = job_name
        self.task_name = task_name
        self.task_info = task_info
        self.cpu = task_info['cpu']
        self.arrival_time = arrival_time
        self.data = data
        self.parent_pos = parent_pos
        self.data_size = data_size

        self.has_layer = [] # 任务含有Layer的位图
        self.container_id = task_info['container_id']
        self.layer = data.containers[self.container_id]
        for i in range(len(data.layers)):
            if(i in data.containers[self.container_id]):
                self.has_layer.append(1)
            else:
                self.has_layer.append(0)

    def get_arrival_time(self):
        return self.arrival_time
    def get_task_id(self):
        return (self.job_name, self.task_name)
    
    # 用于优先队列比较
    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

# 任务队列，按照任务的到达时间排序

class TaskQueue:
    __slots__ = ['tasks']  # 优化内存使用
    
    def __init__(self):
        self.tasks = []
        heapq.heapify(self.tasks)  # 确保堆属性
        
    def add_task(self, task: Task) -> None:
        """添加任务到队列"""
        heapq.heappush(self.tasks, task)
        
    def get_task(self) -> Optional[Task]:
        """获取最早的任务"""
        return heapq.heappop(self.tasks) if self.tasks else None
    
    def peek(self) -> Optional[Task]:
        """查看最早的任务但不移除"""
        return self.tasks[0] if self.tasks else None
    
    def is_empty(self) -> bool:
        return len(self.tasks) == 0
        
    def __len__(self) -> int:
        return len(self.tasks)


def ceil2(value):
    return math.ceil(value*100)/100

# 按至少0.01的粒度occupy，否则会有数值问题
import portion as P
from typing import List, Tuple
import heapq

class Core:
    def __init__(self, idx):
        self.idx = idx
        self.interval = P.closedopen(0, P.inf)
        self.free_intervals = [(0, P.inf)]  # 最小堆存储空闲区间
        
    def occupy(self, start, end):
        i = P.closedopen(start, end)
        if not self.interval.contains(i):
            assert False, "occupy error"
            
        self.interval = self.interval - i
        self._update_free_intervals(start, end)
    
    # [start, end)
    def _update_free_intervals(self, start, end):
        # 维护最小堆中的空闲区间
        new_intervals = []
        while self.free_intervals:
            s, e = heapq.heappop(self.free_intervals)
            if e <= start or s >= end:
                heapq.heappush(new_intervals, (s, e))
            else:
                if s < start:
                    heapq.heappush(new_intervals, (s, start))
                if e > end:
                    heapq.heappush(new_intervals, (end, e))
        self.free_intervals = new_intervals
        
    def find_est(self, start, size) -> float:
        # O(log n) 复杂度找到最早可用时间
        for s, e in self.free_intervals:
            real_start = max(s, start)
            if e >= real_start + size:
                return real_start
        assert False, "no available slot"

    def __repr__(self):
        return self.interval.__str__()

    def __str__(self):
        return self.interval.__str__()

    def __iter__(self):
        return self.interval.__iter__()
    

class Storage:
    def __init__(self, size):
        self.capacity = size  # 缓存容量
        self.used = 0        # 已使用空间
        self.cache = OrderedDict()  # {layer_id: (layer_size, download_finish_time)}
    
    '''
        往缓存里添加layer_id表示的缓存块，大小为layer_size
        1. 若缓存存在,则啥事也没有
        2. 若缓存不存在,则添加缓存。需要保证缓存总大小不超过size, 若超出，则先驱逐旧的缓存块，再添加新的缓存块。驱逐的算法为LRU
    '''
    def add(self, layer_id, layer_size, download_finish_time):
        # 如果已存在，直接返回
        if layer_id in self.cache:
            return
        
        # 检查是否需要腾出空间
        while self.used + layer_size > self.capacity and self.cache:
            # 移除最久未使用的缓存
            _, info = self.cache.popitem(last=False)
            removed_size, _ = info
            self.used -= removed_size
            
        # 如果单个layer太大，则不缓存
        if layer_size > self.capacity:
            return
            
        # 添加新缓存
        self.cache[layer_id] = (layer_size, download_finish_time)
        self.used += layer_size
    
    '''
        判断缓存里是否还有layer_id表示的缓存块
    '''
    def contain(self, layer_id):
        return layer_id in self.cache
    
    '''
        获取层的下载完成时间（用户保证层存在）
    '''
    def get_download_finish_time(self, layer_id):
        return self.cache[layer_id][1]
    
    '''
        标记layer_id对应的缓存块命中一次
    '''
    def hit(self, layer_id):
        if layer_id in self.cache:
            # 将命中的项移到末尾（最新使用）
            size = self.cache.pop(layer_id)
            self.cache[layer_id] = size

    def has_layer(self, L):
        res = []
        for i in range(L):
            if self.contain(i):
                res.append(1)
            else:
                res.append(0)
        return res
    
    def get_all_layers(self):
        return set(self.cache.keys())
    
    def clear(self):
        self.used = 0
        self.cache = OrderedDict()

    # 返回剩余空间
    def remain(self):
        return self.capacity - self.used
    

class Machine:
    def __init__(self, idx: int, node_info: dict, data: dict):
        self.cpu = node_info['cpu']
        self.storage: Storage = Storage(node_info['storage'])
        self.pull_dealy = node_info['pull_delay']
        self.L = len(data.layers)
        self.core_number = int(node_info['core_number'])
        self.idx = idx
        self.layer_size = data.layers
        self.data = data
        self.reset()

    def reset(self):
        self.download_finish_time = 0
        # self.tasks = []
        # self.task_finish_time = 0
        self.total_download_size = 0
        self.cores = [Core(i) for i in range(self.core_number)]
        self.storage.clear()

    @property
    def has_layer(self):
        return self.storage.has_layer(self.L)

    def getRemainingDownloadTime(self, timestamp: float):
        res = []
        for i in range(self.L):
            # 如果没有下载过或者已经下载完成
            if not self.storage.contain(i) or self.storage.get_download_finish_time(i) <= timestamp:
                res.append(0)
            else:
                res.append(self.storage.get_download_finish_time(i)-timestamp)
        return res
    
    # 判断是否还能容纳此任务
    def isAccommodate(self, task: Task):
        # if self.total_download_size + self.getAddLayersSize(task) > self.storage:
        #     return False
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
   
    def addTask(self, task: Task) -> Tuple[float, float]:
        
        timestamp = task.get_arrival_time()
        parent_pos = task.parent_pos
        data_size = task.data_size
        data_ready_time = timestamp + self.data.delay_matrix[parent_pos][self.idx] * data_size

        # self.tasks.append(task)
        add_layers = self.getAddLayers(task)
        self.addNewLayers(add_layers)

        # ready_time为数据准备好，并且镜像层也准备好
        ready_time =ceil2(max(data_ready_time, self.download_finish_time))
        
        # 计算Task完成时间
        execute_time = ceil2(task.cpu / self.cpu)
        core_id, est = self.findEstByCore(ready_time, execute_time)
        self.place(core_id, est, est+execute_time)
        # print(f"{timestamp:.2f}: executing task at [{est:.2f}-{est+execute_time:.2f}) in edge {self.idx}")
        # self.task_finish_time = max(self.download_finish_time, self.task_finish_time) + execute_time
        return est, est + execute_time

    # 添加新的layers，并更新下载完成时间
    def addNewLayers(self, add_layers):
        # 计算Layer下载完成时间
        for layer_id in add_layers:
            self.download_finish_time += self.layer_size[layer_id] * self.pull_dealy
            self.storage.add(layer_id, self.layer_size[layer_id], self.download_finish_time)
            # 记录信息
            self.total_download_size += self.layer_size[layer_id]


    def getAddLayers(self, task: Task):
        # 计算Layer下载完成时间
        layers = set(task.layer)
        add_layers =  layers - self.storage.get_all_layers()
        return add_layers
    
    def getAddLayersSize(self, task: Task):
        add_layers = self.getAddLayers(task)
        return sum(add_layers)
