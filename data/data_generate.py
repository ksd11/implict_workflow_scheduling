import pandas as pd
import numpy as np
from scipy.stats import zipf, expon
import networkx as nx
import os
from .util import visualize_dag


'''
DataGenerator
生成调度的数据集，包含如下信息：
1. Node: N台边缘服务器(N)和一台云服务器, 每台服务器的信息:(容器层传输延迟d_n, 存储s_n, 计算c_n, 核数 core)。假设边缘服务器之间单位传输延迟固定为1, 边缘和云服务器之间传输延迟为15

2. Job: 一个job由多个task组成,task之间有数据传输依赖,组成一个dag.同一个job中有数据传输依赖的task, 传输依赖数据大小随机生成

3. 每个task绑定一个运行环境, 即某种类型的container。task的信息:(job_id, container_id, cpu)

4. Layer: 有多少种Layer(L), 每个layer信息:(layer_size)

5. Container: 共有多少种Container, 每个container信息:(含有的layer信息)。每个container由多个layer组成, 每个contaienr含有5~20个layer, 依赖的layer随机生成

6. Trace: [request1, request2, ..]
    trace长度为Len
    每个request的信息为: (timestamp, job_id)
    其中timestamp之间的间隔通过指数分布生成
        
此外,类提供如下接口
1. generate(job_csv, ...), 生成上述1-6数据,只有job的信息从csv文件读取，其他所需信息可让用户作为参数传入
2. load(path: str), 从一个目录path加载上述1-6所示数据
3. save(path: str), 将1-6所示数据保存到目录path
4. getNewTrace(len, seed=0),返回一条新的trace, 当seed和len相同时, 返回的trace需要一样
5. getSystemInfo(),返回整个系统的信息, 包括上述的1-6点
'''

'''

DCR
LFR


'''
class Config:
    def __init__(self):
        # 机器的信息
        self._edge_delay = 1         # edge和edge之间数据传输延迟
        self.cloud_delay = 15        # edge和cloud之间数据传输延迟
        self._gamma = 1      # average layer pulling latency, cloud is 0.5
        self.lo_storage = 10
        self.hi_storage = 30
        self._c = 1          # 单核平均计算能力
        self.core_number = list(range(1,5))

        # 请求信息
        self._func_comp = 2
        self.request_interval = 1  # 请求到达的间隔
        self._d = 0.5        # average data transmission size

        # layer信息
        self.lo_layer_size = 0.1
        self.hi_layer_size = 2
        self.lo_func_layer_number = 5
        self.hi_func_layer_number = 20

        # cluster相关
        self.num_edge_server = 5
        self.num_layers = 1000
        self.num_containers = 500
        self.trace_len = 100

# 创建全局配置对象
config = Config()

class DataGenerator:
    def __init__(self):
        """初始化数据生成器"""
        self.nodes = []         # 服务器信息
        self.jobs = {}          # 作业信息
        self.layers = []        # 层信息
        self.containers = []    # 容器信息
        self.task_containers = {}  # 任务-容器映射
        self.traces = []        # 请求序列
        
    def generate(self, job_csv
                 , num_edge_nodes=config.num_edge_server
                 , num_layers=config.num_layers
                 , num_containers=config.num_containers
                 , trace_len=config.trace_len
                 , mean_interarrival=config.request_interval
                 , seed=42, zipf_a = 1.5):
        """生成所有需要的数据"""
        np.random.seed(seed)
        
        # 1. 生成节点信息
        self._generate_nodes(num_edge_nodes)
        
        # 2. 读取job信息并添加数据传输大小
        self._load_jobs(job_csv)
        
        # 3. 生成layer信息
        self._generate_layers(num_layers)
        
        # 4. 生成container信息
        self._generate_containers(num_containers)
        
        # 5. 生成task信息
        self._generate_tasks_info(zipf_a = zipf_a)
        
        # 6. 生成请求序列
        self.traces = self.getNewTrace(seed=seed, mean_interarrival=mean_interarrival, trace_len=trace_len)
        
        return self
    
    def _generate_nodes(self, num_edge_nodes):
        """生成边缘节点和云节点信息"""
        n = num_edge_nodes + 1
        self.nodes = [0] * (n)
        self.delay_matrix = np.zeros((n,n))

        # 生成边缘节点间延迟
        for i in range(num_edge_nodes):
            for j in range(i+1, num_edge_nodes):
                delay = np.random.uniform(0.5*config._edge_delay, 1.5*config._edge_delay)
                self.delay_matrix[i][j] = delay
                self.delay_matrix[j][i] = delay
                
        # 设置边缘到云的延迟
        for i in range(num_edge_nodes):
            self.delay_matrix[i][n-1] = config.cloud_delay
            self.delay_matrix[n-1][i] = config.cloud_delay

        # 生成边缘节点
        for i in range(num_edge_nodes):
            self.nodes[i] = {
                'pull_delay': np.random.uniform(0.5*config._gamma, 1.5*config._gamma),
                'storage': np.random.uniform(config.lo_storage, config.hi_storage),  # 随机存储容量
                'cpu': np.random.uniform(0.5*config._c, 1.5*config._c),     # 随机计算能力
                'core_number': np.random.choice(config.core_number)
            }
        
        # 生成云节点
        self.nodes[num_edge_nodes] = {
            'pull_delay': 0.5*config._gamma,
            'storage': config.hi_storage * 2,   # 较大存储容量
            'cpu':  2,                    # 较大计算能力
            'core_number': config.core_number[-1]
        }
    
    def _load_jobs(self, job_csv):
        """加载作业信息并添加数据传输大小和边的概率"""
        df = pd.read_csv(job_csv)
        
        # 按job分组处理
        for job_name in df['job_name'].unique():
            job_df = df[df['job_name'] == job_name]
            
            # 构建DAG
            G = nx.DiGraph()
            dependencies = {}
            parent_children = {}  # 存储父节点到子节点的映射
            
            # 添加节点和解析依赖
            for _, row in job_df.iterrows():
                task_name = row['task_name']
                G.add_node(task_name)
                
                parts = task_name.split('_')
                if len(parts) > 1:
                    deps = [dep for dep in parts[1:] if dep.isdigit()]
                    dependencies[task_name] = deps
            
            # 首先收集所有父节点-子节点关系
            for task, deps in dependencies.items():
                for dep in deps:
                    for pred_task in G.nodes():
                        if pred_task.split('_')[0][-1] == dep:
                            if pred_task not in parent_children:
                                parent_children[pred_task] = []
                            parent_children[pred_task].append(task)
            
            # 为每个父节点的出边分配概率
            for parent, children in parent_children.items():
                num_children = len(children)
                if num_children > 0:
                    # 生成随机概率，确保和为1
                    probabilities = np.random.dirichlet(np.ones(num_children))
                    
                    # 添加边，包含概率和数据大小信息
                    for child, prob in zip(children, probabilities):
                        G.add_edge(parent, child, 
                                 probability=prob,
                                 data_size=np.random.uniform(0.5*config._d, 1.5*config._d))
            
            self.jobs[job_name] = G
            self._add_virtual_source_and_sink(G, job_name)
    
    def _add_virtual_source_and_sink(self, G, job_name):
        """添加虚拟source和sink节点，并设置相应的边概率"""
        # 添加虚拟source节点
        source_node = f"{job_name}_source"
        G.add_node(source_node, cpu=0)  # 虚拟节点不消耗资源
        
        # 添加虚拟sink节点
        sink_node = f"{job_name}_sink"
        G.add_node(sink_node, cpu=0)  # 虚拟节点不消耗资源
        
        # 找出所有入度为0的节点
        start_nodes = [node for node in G.nodes() 
                      if node not in [source_node, sink_node] and G.in_degree(node) == 0]
        
        # 找出所有出度为0的节点
        end_nodes = [node for node in G.nodes() 
                    if node not in [source_node, sink_node] and G.out_degree(node) == 0]
        
        # 为source节点的出边分配概率
        if start_nodes:
            probabilities = np.random.dirichlet(np.ones(len(start_nodes)))
            for node, prob in zip(start_nodes, probabilities):
                G.add_edge(source_node, node, 
                          probability=prob,
                          data_size=0)  # 虚拟边的数据传输大小为0
        
        # 为指向sink节点的边分配概率（每个end节点到sink的概率为1，因为没有其他选择）
        for node in end_nodes:
            G.add_edge(node, sink_node, 
                      probability=1.0,
                      data_size=0)  # 虚拟边的数据传输大小为0

    def _generate_layers(self, num_layers):
        """生成层信息"""
        self.layers = [0] * num_layers
        for i in range(num_layers):
            self.layers[i] = np.random.uniform(config.lo_layer_size, config.hi_layer_size)  # MB
            
    
    def _generate_containers(self, num_containers):
        """生成容器信息"""
        layer_ids = list(range(len(self.layers)))
        self.containers = [0] * num_containers
        
        for i in range(num_containers):
            # 随机选择5-20个层
            num_layers = np.random.randint(config.lo_func_layer_number, config.hi_func_layer_number+1)
            selected_layers = np.random.choice(layer_ids, size=num_layers, replace=False)
            
            self.containers[i] = list(selected_layers)
        
        self.containers.append([]) # -1为空，为source和sink指向的容器

    
    def _generate_tasks_info(self, zipf_a = 1.2):
        """为任务分配容器并生成CPU占用"""
        all_tasks = []
        task_info = {}  # 存储任务的所有信息
        
        # 收集所有任务
        for job_name, G in self.jobs.items():
            for task in G.nodes():
                all_tasks.append((job_name, task))
                if not task.endswith('_source') and not task.endswith('_sink'):  # 排除虚拟节点
                    # 生成随机的CPU占用
                    task_info[(job_name, task)] = {
                        'cpu': np.random.uniform(0.5*config._func_comp, 1.5*config._func_comp)
                    }
        
        container_ids = list(range(len(self.containers)-1)) # 排除最后一个container id
        n = len(container_ids)
    
        # 生成 Zipf 分布的权重
        alpha = zipf_a  # Zipf 分布的参数
        x = np.arange(1, n + 1)
        weights = 1 / (x ** alpha)
        weights /= weights.sum()  # 归一化权重
        
        # 使用 Zipf 分布为每个任务分配容器
        container_assignments = np.random.choice(
            container_ids,
            size=len(all_tasks),
            p=weights,  # 使用 Zipf 权重
            replace=True  # 允许重复使用容器
        )
        
        # 创建任务到容器的映射，同时包含CPU信息
        self.tasks_info = {}
        for (job_name, task), container_id in zip(all_tasks, container_assignments):
            if not task.endswith('_source') and not task.endswith('_sink'):
                self.tasks_info[(job_name, task)] = {
                    'container_id': container_id,
                    'cpu': task_info[(job_name, task)]['cpu']
                }
            else:
                self.tasks_info[(job_name, task)] = {
                    'container_id': -1,
                    'cpu': 0
                }
    
    def save(self, path):
        """保存所有生成的数据"""
        os.makedirs(path, exist_ok=True)
        
        # 保存节点信息
        pd.DataFrame([{'node_id': k, **v} for k, v in enumerate(self.nodes)]).to_csv(
            f'{path}/nodes.csv', index=False)
        
        # 保存作业信息
        job_data = []
        for job_name, G in self.jobs.items():
            for u, v, data in G.edges(data=True):
                job_data.append({
                    'job_name': job_name,
                    'source_task': u,
                    'target_task': v,
                    'data_size': data['data_size'],
                    'probability': data.get('probability', 1.0)  # 添加概率信息，默认为1.0
                })
        pd.DataFrame(job_data).to_csv(f'{path}/jobs.csv', index=False)
        
        # 保存层信息
        pd.DataFrame([{'layer_id': i, 'size': size} for i, size in enumerate(self.layers)]).to_csv(
            f'{path}/layers.csv', index=False)
        
        # 保存容器信息
        container_data = []
        for container_id, layers in enumerate(self.containers):
            container_data.append({
                'container_id': container_id,
                'layers': ','.join(map(str, layers))
            })
        pd.DataFrame(container_data).to_csv(f'{path}/containers.csv', index=False)
        
        # 保存任务信息
        tasks_info_data = []
        for (job_name, task), info in self.tasks_info.items():
            tasks_info_data.append({
                'job_name': job_name,
                'task_name': task,
                'container_id': info['container_id'],
                'cpu': info['cpu']
            })
        pd.DataFrame(tasks_info_data).to_csv(f'{path}/tasks_info.csv', index=False)
        
        # 保存请求序列
        pd.DataFrame(self.traces, columns=['timestamp', 'job_id']).to_csv(
            f'{path}/traces.csv', index=False)
        
        # 将延迟矩阵保存为CSV
        n_nodes = len(self.nodes)
        node_ids = [f'node_{i}' for i in range(n_nodes-1)] + ['cloud']
        delay_df = pd.DataFrame(
            self.delay_matrix,
            index=node_ids,    # 行索引
            columns=node_ids   # 列索引
        )
        delay_df.to_csv(f'{path}/delay_matrix.csv')
    
    def load(self, path):
        """从目录加载数据"""
        # 加载节点信息
        nodes_df = pd.read_csv(f'{path}/nodes.csv')
        self.nodes = []
        self.nodes = [{k: v for k, v in row.items() if k != 'node_id'} for _, row in nodes_df.iterrows()]
        
        # 加载作业信息
        jobs_df = pd.read_csv(f'{path}/jobs.csv')
        self.jobs = {}
        for job_name in jobs_df['job_name'].unique():
            job_edges = jobs_df[jobs_df['job_name'] == job_name]
            G = nx.DiGraph()
            for _, row in job_edges.iterrows():
                G.add_edge(row['source_task'], row['target_task'], data_size=row['data_size'], probability=row['probability'])
            self.jobs[job_name] = G
        
        
        # 3. 加载层信息
        layers_df = pd.read_csv(f'{path}/layers.csv')
        self.layers = [row['size'] for _, row in layers_df.iterrows()]
        
        # 4. 加载容器信息
        containers_df = pd.read_csv(f'{path}/containers.csv')
        self.containers = []
        for _, row in containers_df.iterrows():
            if pd.isna(row['layers']):
                self.containers.append([])
            else:
                self.containers.append(list(map(int, row['layers'].split(','))))
        
        # 5. 加载任务-容器映射
        tasks_info_df = pd.read_csv(f'{path}/tasks_info.csv')
        self.tasks_info = {
            (row['job_name'], row['task_name']): {
                'container_id': row['container_id'],
                'cpu': row['cpu']
            }
            for _, row in tasks_info_df.iterrows()
        }
        
        # 6. 加载请求序列
        traces_df = pd.read_csv(f'{path}/traces.csv')
        self.traces = list(zip(traces_df['timestamp'], traces_df['job_id']))

        # 从CSV加载延迟矩阵
        delay_df = pd.read_csv(f'{path}/delay_matrix.csv', index_col=0)
        self.delay_matrix = delay_df.values

        return self
    
    def getNewTrace(self, seed, trace_len = 100, mean_interarrival = config.request_interval):
        """生成新的请求序列"""
        np.random.seed(seed)
        
        # 生成到达时间间隔（指数分布）
        interarrival_times = np.random.exponential(mean_interarrival, size=trace_len-1)
        timestamps = np.zeros(trace_len)  # 初始化为0
        timestamps[1:] = np.cumsum(interarrival_times)
        
        # 随机选择作业
        job_ids = list(self.jobs.keys())
        selected_jobs = np.random.choice(job_ids, size=trace_len)
        
        return list(zip(timestamps, selected_jobs))
    
    def getSystemInfo(self):
        """返回系统信息"""
        return {
            'nodes': self.nodes,
            'jobs': self.jobs,
            'containers': self.containers,
            'layers': self.layers,
            'tasks_info': self.tasks_info,
            'traces': self.traces
        }
    
    # 在DataGenerator类中添加可视化方法
    def visualize_job(self, job_name, layout_type="hierarchical"):
        """可视化指定作业的DAG"""
        if job_name not in self.jobs:
            raise ValueError(f"Job {job_name} not found")
        
        G = self.jobs[job_name]
        visualize_dag(G, f"Job {job_name} DAG", layout_type)
    
def main():
    # 创建数据生成器
    generator = DataGenerator()
    # generator.load('data/workload_data')
    
    # 生成数据
    generator.generate(
        job_csv='data/selected_jobs.csv',
        num_edge_nodes=config.num_edge_server,
        num_layers=config.num_layers,
        num_containers=config.num_containers,
        trace_len=config.trace_len
    )
    
    # 保存数据
    generator.save('data/workload_data')
    
    # # 打印系统信息
    # print("\n=== 系统信息 ===")
    # print(generator.getSystemInfo())

     # 选择一个作业进行可视化
    job_name = list(generator.jobs.keys())[68]  # 获取第一个作业
    
    # 使用分层布局可视化
    generator.visualize_job(job_name, "hierarchical")
    
    # 使用弹簧布局可视化
    # generator.visualize_job(job_name, "spring")

if __name__ == "__main__":
    main()