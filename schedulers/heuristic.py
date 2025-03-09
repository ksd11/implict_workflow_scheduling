from .scheduler import Scheduler
import numpy as np

# 启发式调度策略，根据状态，统计信息
class HeuristicScheduler(Scheduler):
    def __init__(self, edge_server_num, layer_num):
        self.N = edge_server_num+1
        self.L = layer_num

    '''
        (4 + N) * N  -> (含有的计算资源， 目前存储占用情况， 下载需要等待时间， 需要增量下载的时间, 机器之间通信延迟)
        3     ->  (请求的计算资源， 父任务执行位置， 父任务传输大小)
    '''
    def parse(self, obs):
        node_features = obs[:-3].reshape((4*self.N, self.N))
        # adder_size = [node_feature[3] for node_feature in node_features]
        task_cpu = obs[-3]
        parent_pos = obs[-2]
        data_size = obs[-1]

        download_time = [node_feature[3] for node_feature in node_features]

        wait_time = [node_feature[2] for node_feature in node_features]

        finish_time = []
        for i in range(self.N):
            image_ready_time = wait_time[i] + download_time[i] # 预估
            data_ready_time = node_features[parent_pos][4+i] * data_size
            exec_time = task_cpu / node_features[i][0]
            total_time = max(image_ready_time, data_ready_time) + exec_time
            finish_time.append(total_time)

        return {
            "download_time": download_time,
            "wait_time": wait_time,
            "finish_time": finish_time
        }


    def schedule(self, obs: list)  -> tuple[int, dict]:
        return 0, None