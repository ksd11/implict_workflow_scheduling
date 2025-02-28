
from .scheduler import Scheduler

# 只调度到edge，并且选择下载时间最短的那台机器
class GreedyScheduler(Scheduler):
    def __init__(self, N, L):
        self.N = N
        self.L = L

    def parse_state(self, state):
        N = self.N
        L = self.L
        machine_state = state[:self.N * (3 * self.L + 3)]
        task_state = state[self.N * (3 * self.L + 3):]        

        download_finish_time = []
        for i in range(N):
            download_finish_time.append(
                task_state[i*4+1]
                +task_state[i*4+2])
        return download_finish_time

    def schedule(self, obs: list)  -> tuple[int, dict]:
        download_finish_time = self.parse_state(obs)
        return download_finish_time.index(min(download_finish_time)), None
    