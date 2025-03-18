class Config:
    def __init__(self):
        # 存储所有配置项
        self._config = {
            'job_csv': 'data/selected_jobs.csv',
            '_edge_delay': 1,
            '_gamma': 1,
            'lo_storage': 50,
            'hi_storage': 100,
            '_c': 1,
            'core_number': list(range(1,5)),
            '_func_comp': 2,
            'mean_interarrival': 1,
            '_d': 0.5,
            'lo_layer_size': 0.1,
            'hi_layer_size': 2,
            'lo_func_layer_number': 5,
            'hi_func_layer_number': 20,
            'num_edge_nodes': 5,
            'num_layers': 1000,
            'num_containers': 500,
            'trace_len': 2000
        }
        
    def __getattr__(self, name):
        """支持点访问"""
        return self._config[name]
        
    def __setattr__(self, name, value):
        """支持点赋值"""
        if name == '_config':
            super().__setattr__(name, value)
        else:
            self._config[name] = value
            
    def __getitem__(self, key):
        """支持下标访问"""
        return self._config[key]
        
    def __setitem__(self, key, value):
        """支持下标赋值"""
        self._config[key] = value

    def __str__(self):
        return str(self._config)
    
global_config = Config()