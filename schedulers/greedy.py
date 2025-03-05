
from .scheduler import Scheduler

# 选择下载时间最短的那台机器
class GreedyScheduler(Scheduler):
    def __init__(self, edge_server_num, layer_num):
        self.total_server = edge_server_num + 1
        self.total_layer = layer_num

    def parse_state(self, state):
        total_server = self.total_server
        total_layer = self.total_layer
        # machine_state = state[:self.N * (3 * self.L + 3)]
        # task_state = state[self.N * (3 * self.L + 3):]  
        machine_state = state[:total_server * (total_server + 4)].reshape((total_server,total_server+4))
        # task_state = state[-3:]

        download_finish_time = []
        for i in range(total_server):
            download_finish_time.append(machine_state[i][2] + machine_state[i][3])
            # download_finish_time.append(
            #     task_state[i*4+1]
            #     +task_state[i*4+2])
        return download_finish_time

    def schedule(self, obs: list)  -> tuple[int, dict]:
        download_finish_time = self.parse_state(obs)
        return download_finish_time.index(min(download_finish_time)), None
    