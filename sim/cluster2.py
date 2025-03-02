import math
from typing import Tuple
import portion as P
from collections import OrderedDict
from data.data_generate import DataGenerator

import heapq
from typing import Any, Optional

class Task:
    def __init__(self, job_id: str, task_id :str, task_info: dict, arrival_time: float, data: DataGenerator):
        self.job_id = job_id
        self.task_id = task_id
        self.task_info = task_info
        self.arrival_time = arrival_time
        self.data = data
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
    
    # 用于优先队列比较
    def __lt__(self, other):
        return self.arrival_time < other.arrival_time

# 任务队列，按照任务的到达时间排序
class TaskQueue:
    def __init__(self):
        self.tasks = []  # 优先队列
        
    def add_task(self, task:Task) -> None:
        """添加任务到队列"""
        heapq.heappush(self.tasks, task)
        
    def get_task(self) -> Optional[Task]:
        """获取最早的任务"""
        if not self.tasks:
            return None
        return heapq.heappop(self.tasks)
    
    def peek(self) -> Optional[Task]:
        """查看最早的任务但不移除"""
        if not self.tasks:
            return None
        return self.tasks[0]
    
    def is_empty(self) -> bool:
        return len(self.tasks) == 0


def ceil2(value):
    return math.ceil(value*100)/100

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
    

class Machine:
    def __init__(self, cpu: float, storage: float, bandwidth: float, layer_size: list, core_number:int, idx: int):
        self.cpu = cpu
        self.storage = Storage(storage)
        self.bandwidth = bandwidth
        self.L = len(layer_size)
        self.core_number = core_number
        self.idx = idx
        self.layer_size = layer_size
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
                res.append(timestamp)
            else:
                res.append(self.storage.get_download_finish_time(i))
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
   
    def addTask(self, task: Task, timestamp: float) -> Tuple[float, float]:
        # self.tasks.append(task)
        add_layers = self.getAddLayers(task)
        self.addNewLayers(add_layers)

        # 计算Layer下载完成时间
        ready_time =ceil2(max(timestamp, self.download_finish_time))
        
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
            self.download_finish_time += self.layer_size[layer_id]/self.bandwidth
            self.storage.add(layer_id, self.layer_size[layer_id], self.download_finish_time)
            # 记录信息
            self.total_download_size += self.layer_size[layer_id]


    def getAddLayers(self, task: Task):
        # 计算Layer下载完成时间
        layers = set(task.layer)
        add_layers =  layers - self.storage.get_all_layers()
        return add_layers
    
    def getAddLayersSize(self, task: Task):
        return sum([self.storage.used])

class Cloud(Machine):
    def __init__(self, cpu: float, storage: float, bandwidth: float, layer_size: list):
        super().__init__(cpu, storage, bandwidth, layer_size, 4, -1)

    # def addTask(self, task: Task, timestamp: float) -> Tuple[float, float]:
    #     add_layers = self.getAddLayers(task)
    #     self.addNewLayers(add_layers)

    #     # 计算Layer下载完成时间
    #     ready_time =ceil2(max(timestamp, self.download_finish_time))

    #     # 计算Task完成时间
    #     execute_time = ceil2(task.cpu / self.cpu)
    #     est = ready_time
    #     return est, est + execute_time
    
    # def isAccommodate(self, task: Task):
    #     return True