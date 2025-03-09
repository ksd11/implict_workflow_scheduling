from .heuristic import HeuristicScheduler

# 选择等待时间最短的那台机器
class DepEFTScheduler(HeuristicScheduler):
    def __init__(self, edge_server_num, layer_num):
        super(DepEFTScheduler, self).__init__(edge_server_num, layer_num)

    def schedule(self, obs: list)  -> tuple[int, dict]:
        finish_time = self.parse(obs)["finish_time"]
        return finish_time.index(min(finish_time)), None