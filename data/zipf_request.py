import numpy as np
from scipy.stats import zipf

def generate_zipf_requests(n_types, n_samples, alpha=1.5):
    """
    n_types: 请求类型数量
    n_samples: 需要生成的请求数量
    alpha: Zipf分布参数，值越大分布越偏向热门类型
    """
    # 生成Zipf分布概率
    x = np.arange(1, n_types + 1)
    weights = 1 / (x ** alpha)
    weights = weights / weights.sum()  # 归一化
    
    # 按概率抽样
    requests = np.random.choice(
        np.arange(n_types),  # 请求类型ID: 0 to n_types-1
        size=n_samples,      # 生成数量
        p=weights,           # 每种类型的概率
        replace=True         # 允许重复
    )
    
    return requests

# 使用示例
# n_types = 20    # 20种请求类型
# n_samples = 100 # 生成100个请求
# alpha = 2.0     # Zipf分布参数

# requests = generate_requests(n_types, n_samples, alpha)

# print(requests)