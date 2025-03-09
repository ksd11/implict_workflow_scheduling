from .greedy import GreedyScheduler
import numpy as np
import networkx as nx

'''
根据MLP路径预部署镜像

'''
class XanaduScheduler(GreedyScheduler):
    def __init__(self, env, predeploy_degree = 1):
        self.env = env
        self.predeploy_degree = predeploy_degree
        edge_server_num = env.N
        layer_num = env.L
        super(XanaduScheduler, self).__init__(edge_server_num, layer_num)

    def after_deploy_hook_func(self):
        timestamp, task_name, job_name, G, tasks_info = self.env.get_sched_additional_info()

        # 选择最有可能执行的函数预部署
        task_list:list[str] = []

        while len(task_list) < self.predeploy_degree:
            successors = list(G.successors(task_name))
            if len(successors) == 0:
                break
            probs = np.array([G[task_name][succ].get('probability', 1.0) 
                        for succ in successors])
            chosen_succ = successors[probs.argmax()]
            task_list.append(chosen_succ)
            task_name = chosen_succ

        # 从任务列表获取所有对应的容器
        container_list = [tasks_info[(job_name, tn)]["container_id"] for tn in task_list]
        # 预部署容器
        for container_id in container_list:
            self.env.predeploy(timestamp, container_id)


    def schedule(self, obs: list)  -> tuple[int, dict]:
        download_finish_time = self.parse_state(obs)
        return download_finish_time.index(min(download_finish_time)), self.after_deploy_hook_func