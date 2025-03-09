
from .heuristic import HeuristicScheduler

# 选择等待时间最短的那台机器
class DepWaitScheduler(HeuristicScheduler):
    def __init__(self, edge_server_num, layer_num):
        super(DepWaitScheduler, self).__init__(edge_server_num, layer_num)

    def schedule(self, obs: list)  -> tuple[int, dict]:
        wait_time = self.parse(obs)["wait_time"]
        return wait_time.index(min(wait_time)), None