from .scheduler import Scheduler
import numpy as np

# 只调度到edge，并且选择下载时间最短的那台机器
class RandomScheduler(Scheduler):
    def __init__(self, N, L):
        self.N = N
        self.L = L

    def schedule(self, obs: list)  -> tuple[int, dict]:
        return np.random.choice(self.N+1),None
    