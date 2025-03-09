
from .scheduler import Scheduler
from .heuristic import HeuristicScheduler

# 选择下载时间最短的那台机器
class DepDownScheduler(HeuristicScheduler):
    def __init__(self, edge_server_num, layer_num):
        super(DepDownScheduler, self).__init__(edge_server_num, layer_num)

    def schedule(self, obs: list)  -> tuple[int, dict]:
        download_time = self.parse(obs)["download_time"]
        return download_time.index(min(download_time)), None
    