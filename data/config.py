class Config:
    def __init__(self):
        self.job_csv = 'data/selected_jobs.csv'

        # 机器的信息
        self._edge_delay = 1         # edge和edge之间数据传输延迟
        self.cloud_delay = 15        # edge和cloud之间数据传输延迟
        self._gamma = 1      # average layer pulling latency, cloud is 0.5
        self.lo_storage = 50
        self.hi_storage = 100
        self._c = 1          # 单核平均计算能力
        self.core_number = list(range(1,5))

        # 请求信息
        self._func_comp = 2
        self.mean_interarrival = 1  # 请求到达的间隔
        self._d = 0.5        # average data transmission size

        # layer信息
        self.lo_layer_size = 0.1
        self.hi_layer_size = 2
        self.lo_func_layer_number = 5
        self.hi_func_layer_number = 20

        # cluster相关
        self.num_edge_nodes = 5
        self.num_layers = 1000
        self.num_containers = 500
        self.trace_len = 2000