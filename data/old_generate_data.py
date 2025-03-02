import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from typing import Tuple
from .zipf_request import generate_zipf_requests

'''
Node: 有多少台边缘机器（N）, 每台机器的信息：（带宽b_n, 存储d_n, cpu f_n），以及一台cloud服务器
Layer: 有多少个Layer(L)，每个layer的大小
Container: 共有多少种Container，也即请求的种类数
           每个container信息：（cpu f_c， layer info）

Traces: [trace1, trace2, ..]
其中每条trace:
每条trace长度固定, 假设为Len
然后每个step的信息为：(task_id, timestamp, container_id)
其中timestamp之间的间隔通过指数分布生成，假设两个请求之间间隔符合指数分布
container_id采用均匀分布生成

'''

class Data:
    # 机器的信息
    lo_bandwidth = 0.6 # 60Mbps~90Mbps
    hi_bandwidth = 0.9
    lo_storage = 5  # 小存储
    hi_storage = 10
    lo_cpu = 0.8 # 0.8GHz ~ 1.2GHz
    hi_cpu = 1.2

    # 请求信息
    lo_request_cpu = 0.1
    hi_request_cpu = 2
    request_interval = 1 # 请求到达的间隔

    # layer信息
    lo_layer_size = 0.1
    hi_layer_size = 2
    lo_func_layer_number = 5
    hi_func_layer_number = 20


    # 获取请求的到达时间
    def getRequestArrivals(self, interval, Len):
        intervals = np.random.exponential(interval, Len)
        arrivals = np.zeros(intervals.shape)
        for i in range(1, Len):
            arrivals[i] = arrivals[i-1]+intervals[i-1]
        return arrivals
    
    # 获取请求的container类型
    def getContainerTypes(self, Len):
        # container_types = generate_zipf_requests(self.N, Len, 2)
        container_types = np.random.randint(0, self.C, Len)
        return container_types

    # 获取请求的信息
    def getTrace(self, Len):
        arrivals = self.getRequestArrivals(self.request_interval, Len)
        container_types = self.getContainerTypes(Len)
        task_ids = np.array(range(Len)) # 给每一个task分配一个id
        return np.column_stack((task_ids, arrivals, container_types))
        # print(np.column_stack((arrivals, container_types)))

    # 获取机器信息
    def getMachines(self, N):
        machines = []
        for i in range(N):
            machine = dict()
            machine['cpu'] = np.random.uniform(self.lo_cpu, self.hi_cpu)
            machine['storage'] = np.random.uniform(self.lo_storage, self.hi_storage)
            machine['bandwidth'] = np.random.uniform(self.lo_bandwidth, self.hi_bandwidth)
            machines.append(machine)
        # print(machines)
        return machines
    
    def getLayers(self, L):
        layers = np.zeros(L)
        for i in range(L):
            layers[i] = np.random.uniform(self.lo_layer_size, self.hi_layer_size)
        # print(layers)
        return layers
    
    def getContainerInfo(self, L, C):
        containers = []
        for i in range(C):
            container = dict()
            container['cpu'] = np.random.uniform(self.lo_request_cpu, self.hi_request_cpu)
            layer_number = np.random.randint(self.lo_func_layer_number, self.hi_func_layer_number+1)
            container['layer'] = np.random.choice(np.arange(0, L), layer_number, replace=False)
            container['layer'].sort()
            containers.append(container)
        # print(containers)
        return containers

    def getCloud(self):
        cloud = dict()
        cloud['cpu'] = self.hi_cpu
        cloud['storage'] = self.hi_storage
        cloud['bandwidth'] = 1
        return cloud

    def __init__(self, N, L, C, Len):
        np.random.seed(0) # 保证环境是一样的
        self.N = N # 机器个数
        self.L = L # layer个数
        self.C = C # container个数
        self.Len = Len
        self.machines = self.getMachines(N)
        self.layers = self.getLayers(L)
        self.containers = self.getContainerInfo(L, C)
        self.trace = self.getAnotherTrace()
        self.cloud = self.getCloud()
        # return machines, layers, containers, trace
    
    def getAnotherTrace(self):
        # if not hasattr(self, 'trace'):
        self.trace = self.getTrace(self.Len)
        return self.trace
        # return self.trace

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


'''
N: 边缘服务器个数
L: 容器层个数
C: 容器种类数
Len: 100 请求长度
'''

global_data = Data(N=5, L=100, C=30, Len=1000)