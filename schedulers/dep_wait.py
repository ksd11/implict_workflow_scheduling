
from .heuristic import HeuristicScheduler
import numpy as np

# 选择等待时间最短的那台机器
class DepWaitScheduler(HeuristicScheduler):
    def __init__(self, edge_server_num, layer_num):
        super(DepWaitScheduler, self).__init__(edge_server_num, layer_num)

    def schedule(self, obs: list)  -> tuple[int, dict]:
        wait_time = self.parse(obs)["wait_time"]
        # return wall_time.index(min(wait_time)), None
        
        wait_time = np.array(wait_time)
        
        # 1. 找到最小值的所有位置
        min_indices = np.where(wait_time == wait_time.min())[0]
        
        # 2. 随机选择一个索引
        return np.random.choice(min_indices), None