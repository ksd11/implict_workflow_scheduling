import math
from typing import Tuple
from data.data_generate import DataGenerator
from .storage import Storage,FCFSStorage,LRUStorage,PriorityPlusStorage,PriorityStorage

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
        gen_pos: 只针对source和sink的虚拟任务
    '''
    def __init__(self, job_name: str, task_name :str, arrival_time: float, data: DataGenerator, parent_pos: int = 0, data_size: float = 0, origin_pos = -1, global_id = -1):
        task_info = data.tasks_info[(job_name,task_name)]
        self.job_name = job_name
        self.task_name = task_name
        self.task_info = task_info
        self.cpu = task_info['cpu']
        self.arrival_time = arrival_time
        self.data = data
        self.parent_pos = parent_pos
        self.data_size = data_size
        self.origin_pos = origin_pos
        self.global_id = global_id

        self.has_layer = [] # 任务含有Layer的位图
        self.container_id = task_info['container_id']
        self.layer = set(data.containers[self.container_id])
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
    

class Machine:
    def __init__(self, idx: int, node_info: dict, data: dict):
        self.cpu = node_info['cpu']
        self.storage: Storage = FCFSStorage(node_info['storage'])
        # self.storage: Storage = PriorityStorage(node_info['storage'])
        # self.storage: Storage = PriorityPlusStorage(node_info['storage'])

        self.pull_dealy = node_info['pull_delay']
        self.L = len(data.layers)
        self.core_number = int(node_info['core_number'])
        self.idx = idx
        self.layer_size = data.layers
        self.data = data
        self.reset()

    def reset(self):
        self.download_finish_time = 0
        self.total_download_size = 0
        self.data_transmission_time = 0 # 数据传输时间
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
    
    def findEstByCore(self, start, size):
        res = math.inf
        core_id = -1
        for idx, core in enumerate(self.cores):
            est = core.find_est(start, size)
            if res > est:
                res = est
                core_id = idx
        return core_id, res
    
    def get_image_ready_time(self, layers):
        max_download_finish_time = 0
        for layer in layers:
            max_download_finish_time = max(max_download_finish_time, self.storage.get_download_finish_time(layer))
        return max_download_finish_time

    def place(self, core_id, start, end):
        # print(f"edge[{self.idx}-{core_id}] occupy: {start}-{end}")
        self.cores[core_id].occupy(start, end)
   
    def addTask(self, task: Task) -> Tuple[float, float]:
        
        timestamp = task.get_arrival_time()
        parent_pos = task.parent_pos
        data_size = task.data_size

        # 数据传输时间
        data_tranmission = self.data.delay_matrix[parent_pos][self.idx] * data_size
        data_ready_time = timestamp + data_tranmission
        self.data_transmission_time += data_tranmission

        # self.tasks.append(task)
        add_layers = self.getAddLayers(task.layer, hit=True) # 设置hit标记
        if len(add_layers) == 0:
            # 镜像层全部命中
            image_ready_time = self.get_image_ready_time(add_layers)
        else:
            # 需要下载部分镜像层
            self.addNewLayers(timestamp=timestamp, add_layers=add_layers)
            image_ready_time = self.download_finish_time

        # ready_time为数据准备好，并且镜像层也准备好
        ready_time =ceil2(max(data_ready_time, image_ready_time))
        
        # 计算Task完成时间
        execute_time = ceil2(task.cpu / self.cpu)
        # core_id, est = self.findEstByCore(ready_time, execute_time)
        # self.place(core_id, est, est+execute_time)
        core_id, est = 0, ready_time
        return est, est + execute_time

    # 添加新的layers，并更新下载完成时间
    def addNewLayers(self, timestamp, add_layers):
        # 不能在timestamp之前下载，因为决策是timestamp时做的
        self.download_finish_time = max(timestamp, self.download_finish_time)
        # 计算Layer下载完成时间
        for layer_id in add_layers:
            self.download_finish_time += self.layer_size[layer_id] * self.pull_dealy
            self.storage.add(layer_id, self.layer_size[layer_id], self.download_finish_time)
            # 记录信息
            self.total_download_size += self.layer_size[layer_id]


    # hit: 是否要标记使用的容器层
    def getAddLayers(self, layers, hit = False):
        # 计算Layer下载完成时间
        # layers = task.layer
        add_layers =  layers - self.storage.get_all_layers()

        # 若标记了需要设置命中
        if hit:
            hit_layers = layers - add_layers
            for layer_id in hit_layers:
                self.storage.hit(layer_id)
        
        return add_layers
    
    def getAddLayersSize(self, layers):
        add_layers = self.getAddLayers(layers)
        return sum([self.layer_size[layer] for layer in add_layers])

    # 在时刻timestamp预部署容器(真正部署)
    def predeploy_container(self, timestamp, container_id):
        # 获取容器对应的所有容器层
        layers = set(self.data.containers[container_id])
        add_layers = self.getAddLayers(layers)
        self.addNewLayers(timestamp, add_layers)

    # 假如在timestamp需要拉取容器，则拉取完毕的时间 (不是真正的部署，只是预估)
    def get_ready_time_of_container(self, timestamp, container_id):
        layers = set(self.data.containers[container_id])
        add_layers = self.getAddLayers(layers)
        res = max(timestamp, self.download_finish_time)
        # 计算Layer下载完成时间
        for layer_id in add_layers:
            res += self.layer_size[layer_id] * self.pull_dealy
        return res
