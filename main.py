
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from cfg_loader import load
from pprint import pprint
import sim
from schedulers import scheduler_mapping, Scheduler
import numpy as np

# 防止中文乱码
import matplotlib.pyplot as plt
font_name = "simhei"
plt.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
plt.rcParams['axes.unicode_minus']=False # 正确显示负号，防止变成方框

def make_parser():
    parser = ArgumentParser(
        description=__doc__, formatter_class=ArgumentDefaultsHelpFormatter
    )

    # 指定scheduler的名字
    parser.add_argument(
        "-s",
        "--scheduler",
        dest="scheduler",
        help="scheduler name",
        required=True,
    )

    # 指定配置文件的路径
    parser.add_argument(
        "-f",
        "--file",
        dest="filename",
        help="experiment definition file",
        metavar="FILE",
        required=True,
    )

    return parser

# env = sim.LayerEdgeEnv()
env = sim.LayerEdgeDynamicEnv()
scheduler = {
    "dep-down": {
        "edge_server_num": env.N,
        "layer_num": env.L
    }, 
    "dep-wait": {
        "edge_server_num": env.N,
        "layer_num": env.L
    }, 
    "dep-eft": {
        "edge_server_num": env.N,
        "layer_num": env.L
    }, 
    "random": {
        "edge_server_num": env.N,
        "layer_num": env.L
    },
     "dqn":{
        "config_path": "config/dqn.yaml"
    },
    "ppo":{
        "config_path": "config/ppo.yaml"
    }, 
    # "xanadu": {
    #     "env": env,
    #     "predeploy_degree": 1
    # }
    }

def report(info:dict, verbose = False):
    makespan = max([info[k]['finish_time'] for k in info])
    
    # 请求的结束时间其sink函数的结束时间
    # all_request_finish_time = {info[k]['global_id']: info[k]['finish_time'] for k in info if info[k]['task_id'][1].endswith("sink")}
    # all_request_arrival_time = {info[k]['global_id']: info[k]['arrival_time'] for k in info if info[k]['task_id'][1].endswith("source")}
    # all_request_process_time = {k: all_request_finish_time[k] - all_request_arrival_time[k] for k in all_request_finish_time}

    if verbose:
        for k in info:
            print(info[k])
        print("Makespan is: ", makespan)

    return {
        'makespan': makespan,
        # 'all_request_process_time': all_request_process_time
    }

def one_experiment(env, scheduler: Scheduler, seed = None, options = {'trace_len': 100}, verbose = False):

    state,_ = env.reset(seed=seed, options=options)
    reward_sum = 0
    done = False

    while not done:
        action ,_func = scheduler.schedule(state)
        state, reward, terminated, truncated, info = env.step(int(action), after_deploy_hook_func = _func)
        done = terminated or truncated
        reward_sum += reward
        env.render()

    env.close()
    # pprint(info)
    statistics = report(info['schedule_info'], verbose=verbose)
    # return reward_sum
    return {
        "reward_sum": reward_sum,
        **statistics
    }


# 根据task_number变化的折线图
def plot_results(results: dict, x_values: list, title: str = "算法对比"):
    plt.figure(figsize=(10, 6))
    
    # 为每个算法画一条线
    for algo_name, values in results.items():
        plt.plot(x_values, values, marker='o', label=algo_name)
    
    # 设置图表属性
    plt.title(title)
    plt.xlabel('请求数量')
    plt.ylabel('完成时间')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    plt.show()
    plt.savefig('comparison.png')
    plt.close()


import json
def comparation():
    results = {}
    request_len_array = [100,200,400,600,800,1000]
    for sched, info in scheduler.items():
        schedulerCls = scheduler_mapping[sched](**info)
        results[sched] = []
        for trace_len in request_len_array:
            info = one_experiment(env=env, scheduler=schedulerCls, seed=0, options={'trace_len': trace_len})
            results[sched].append(info["makespan"])
            print(f"scheduler: {sched}")
            print(f"trace_len: {trace_len}")
            print(f"info: {info}")
            print()

    pprint(results)

    # 使用示例
    results = {
        "dqn": results["dqn"],
        "ppo": results["ppo"],
        "random": results["random"],
        "dep-down": results["dep-down"],
        "dep-wait": results["dep-wait"],
        "dep-eft": results["dep-eft"],
    }

    # 保存为JSON文件
    with open('results.json', 'w') as f:
        json.dump(results, f, indent=4)

    plot_results(results, request_len_array, "不同算法在不同请求数量下的完成时间对比")


def xanadu_different_predeploy_degree():
    results = {}
    request_len_array = [100,200,400,600,800,1000]
    predeploy_degree = range(1,7)
    scheduler_info = scheduler["xanadu"]

    for degree in predeploy_degree:
        sched = f"xanadu-{degree}"
        scheduler_info["predeploy_degree"] = degree

        schedulerCls = scheduler_mapping["xanadu"](**scheduler_info)
        results[sched] = []
        for trace_len in request_len_array:
            info = one_experiment(env=env, scheduler=schedulerCls, seed=0, options={'trace_len': trace_len})
            results[sched].append(info["makespan"])
            print(f"scheduler: {sched}")
            print(f"trace_len: {trace_len}")
            print(f"info: {info}")
            print()

    pprint(results)

    # 使用示例
    results_dict = {}
    for degree in predeploy_degree:
        sched = f"xanadu-{degree}"
        results_dict[sched] = results[sched]

    plot_results(results_dict, request_len_array, "不同算法在不同请求数量下的完成时间对比")

def test0():
    scheduler_name = "dep-eft"
    sched = scheduler_mapping[scheduler_name](**scheduler[scheduler_name])
    info = one_experiment(env=env, scheduler=sched, seed=0, options={'trace_len': 20}, verbose=True)
    print(info)


def plot_cdf(results: dict):
    plt.figure(figsize=(10, 6))
    
    for scheduler_name, finish_times_list in results.items():
        # 获取完成时间数组
        finish_times = finish_times_list  # 取第一个实验结果
        
        # 计算CDF
        sorted_data = np.sort(finish_times)
        p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF线
        plt.plot(sorted_data, p, label=scheduler_name)
    
    # 设置图表属性
    plt.xlabel('完成时间')
    plt.ylabel('累积概率')
    plt.title('不同调度策略的请求完成时间CDF对比')
    plt.grid(True)
    plt.legend()
    
    # 保存图表
    plt.savefig('cdf.png')
    plt.close()

    
# 每种调度器执行1000个请求，然后画出各自的完成时间cdf图
def cdf(seed = 0):
    results = {}
    trace_len = 1000
    for sched, info in scheduler.items():
        schedulerCls = scheduler_mapping[sched](**info)
        info = one_experiment(env=env, scheduler=schedulerCls, seed=seed, options={'trace_len': trace_len})
        results[sched] = info["all_request_finish_time"]
        print(f"scheduler: {sched}")
        print(f"trace_len: {trace_len}")
        view = info["all_request_finish_time"][:10]
        print(f"all_request_finish_time: {view}...")
        print()

    # 保存为JSON文件
    with open('__result__/cdf.json', 'w') as f:
        json.dump(results, f, indent=4)

    # pprint(results)
    plot_cdf(results)


if __name__ == "__main__":
    # comparation()
    test0()
    # xanadu_different_predeploy_degree()

    # cdf()
    
