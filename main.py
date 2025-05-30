
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from cfg_loader import load
from pprint import pprint
import sim
from schedulers import scheduler_mapping, Scheduler
import numpy as np
import pandas as pd
from sim.storage import FCFSStorage, LRUStorage, PopularityStorage, PriorityBigStorage, PriorityStorage

# 防止中文乱码
import matplotlib.pyplot as plt
font_name = "simhei"
plt.rcParams['font.family']= font_name # 指定字体，实际上相当于修改 matplotlibrc 文件　只不过这样做是暂时的　下次失效
plt.rcParams['axes.unicode_minus']=False # 正确显示负号，防止变成方框

# 1. 全局字体大小设置
plt.rcParams.update({
    'font.size': 24,              # 基础字体大小
    'axes.labelsize': 24,         # 坐标轴标签字体大小
    'axes.titlesize': 24,         # 标题字体大小
    'xtick.labelsize': 22,        # x轴刻度标签字体大小
    'ytick.labelsize': 22,        # y轴刻度标签字体大小
    'legend.fontsize': 24,        # 图例字体大小
    'figure.titlesize': 26        # 图表标题字体大小
})
    

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
env = sim.LayerEdgeDynamicEnv(need_log=True)
scheduler = {
    "Dep-Down": {
        "edge_server_num": env.N,
        "layer_num": env.L
    }, 
    "Dep-Wait": {
        "edge_server_num": env.N,
        "layer_num": env.L
    }, 
    "Dep-Eft": {
        "edge_server_num": env.N,
        "layer_num": env.L
    }, 
    "Random": {
        "edge_server_num": env.N,
        "layer_num": env.L
    },
    "DQN":{
        "config_path": "config/dqn.yaml"
    },
    "PPO":{
        "config_path": "config/ppo.yaml"
    }, 
    # "xanadu": {
    #     "env": env,
    #     "predeploy_degree": 1
    # }
    }

colors = {
        'Random': '#2ca02c',   # 绿色
        'Dep-Down': '#1f77b4', # 蓝色
        'Dep-Eft': '#9467bd',  # 紫色
        'Dep-Wait': '#8c564b',  # 棕色
        'DQN': '#ff7f0e',      # 橙色
        'PPO': '#d62728',      # 红色
    }

markers = {
    'DQN': 'o',      # 圆形
    'PPO': 's',      # 方形
    'Random': '^',   # 上三角
    'Dep-Down': 'D', # 菱形
    'Dep-Eft': 'v',  # 下三角
    'Dep-Wait': 'p'  # 五角星
}

linestyles = {
    'PPO': 'solid',                    # 实线
    'DQN': (0, (5, 1)),               # 长虚线         
    'Dep-Wait': (0, (3, 1, 1, 1)),    # 短线-点-点 
    'Dep-Down': 'dotted',              #点线
    'Random':  'dashed',             # 虚线
    'Dep-Eft': 'dashdot',            # 点划线
}

def report(infos:dict, verbose = False):
    tasks_execution_info = infos["tasks_execution_info"]
    machines_info = infos["machines_info"]

    makespan = max([info['finish_time'] for info in tasks_execution_info])
    
    # tasks_execution_info 请求的结束时间其sink函数的结束时间
    all_request_finish_time = {info['global_id']: info['finish_time'] for info in tasks_execution_info if info['task_id'][1].endswith("sink")}
    all_request_arrival_time = {info['global_id']: info['arrival_time'] for info in tasks_execution_info if info['task_id'][1].endswith("source")}
    all_request_process_time = {k: all_request_finish_time[k] - all_request_arrival_time[k] for k in all_request_finish_time}

    all_task_waiting_time = [info['start_time']-info['arrival_time'] for info in tasks_execution_info]

    all_task_execution_time = [info['finish_time'] - info['start_time'] for info in tasks_execution_info]

    all_task_wait_for_image = [info['wait_for_image'] for info in tasks_execution_info]
    all_task_wait_for_data = [info['wait_for_data'] for info in tasks_execution_info]
    all_task_wait_for_comp = [info['wait_for_comp'] for info in tasks_execution_info]

    # machines_info解析
    all_machine_download_time = [v["download_finish_time"] for v in machines_info]
    all_machine_download_size = [v["total_download_size"] for v in machines_info]
    
    machine_execute_task_num = [0] * len(machines_info)
    for info in tasks_execution_info:
        machine_execute_task_num[info['server_id']]+=1

    # all_machine_data_tranmission_time = [v["data_transmission_time"] for v in machines_info]

    if verbose:
        for info in infos:
            print(info)
        print("Makespan is: ", makespan)

    return {
        'makespan': makespan,
        'all_request_process_time': all_request_process_time,
        'all_task_execution_time': all_task_execution_time,

        'all_machine_download_time': all_machine_download_time,
        'all_machine_download_size': all_machine_download_size,
        # 'all_machine_data_tranmission_time': all_machine_data_tranmission_time,

        'all_task_waiting_time': all_task_waiting_time,
        'all_task_wait_for_image': all_task_wait_for_image,
        'all_task_wait_for_data': all_task_wait_for_data,
        'all_task_wait_for_comp': all_task_wait_for_comp,

        'machine_execute_task_num': machine_execute_task_num
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
def plot_results(results: dict, x_values: list, x_label: str = "请求数量", y_label="完成时间",fig_name = "comparison", algos = ["PPO", "DQN", "Dep-Wait", "Dep-Eft", "Random", "Dep-Down"], threshold: float = 100000, legend_pos = "best"):
    plt.figure(figsize=(8, 6))

    if algos is None:
        algos = list(results.keys())

    x_outliers = {}

    for algo_name in algos:
        values = results[algo_name]
        y = np.array(values)
        x = np.array(x_values)
        
        over_threshold = y > threshold
        normal_points = ~over_threshold
        
        color = colors.get(algo_name, '#1f77b4')
        line = plt.plot(x[normal_points], y[normal_points], 
                       marker=markers.get(algo_name, 'o'),
                       color=color,
                       linestyle = linestyles.get(algo_name, 'solid'),
                       linewidth = 3,
                       label=algo_name)[0]
        
        if any(over_threshold):
            for xi, yi in zip(x[over_threshold], y[over_threshold]):
                # 计算垂直偏移
                if xi not in x_outliers:
                    x_outliers[xi] = 0
                y_offset = x_outliers[xi] * (-10000)  # 向下偏移
                x_outliers[xi] += 1
                
                # 绘制偏移后的点
                plt.scatter(xi, threshold + y_offset,
                          marker=markers.get(algo_name, 'o'),
                          s=100,
                          color=color)
                
                # 标注也相应偏移
                plt.annotate(f'{int(yi)}', 
                           (xi, threshold + y_offset),
                           xytext=(0, -20),
                           textcoords='offset points',
                           ha='right',
                           va='center',
                           fontsize=15,
                           color=color)
    
    # plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.grid(True)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc=legend_pos)
    
    plt.savefig(fig_name+".pdf", bbox_inches='tight', dpi=300)
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
    info = one_experiment(env=env, scheduler=sched, seed=0, options={'trace_len': 1000}, verbose=False)
    # print(sum(info['all_request_process_time'].values()))


def plot_cdf(results: dict, algos = ["PPO","DQN", "Dep-Wait", "Dep-Eft"]):
    plt.figure(figsize=(8, 6))
    
    for scheduler_name in algos:
        finish_times_list = results[scheduler_name]
        # 获取完成时间数组
        finish_times = finish_times_list  # 取第一个实验结果
        
        # 计算CDF
        sorted_data = np.sort(finish_times)
        p = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
        
        # 绘制CDF线
        plt.plot(sorted_data, p, label=scheduler_name, color=colors[scheduler_name], linewidth=3)
    
    # 设置图表属性
    plt.xlabel('完成时间')
    plt.ylabel('累积概率')
    # plt.title('不同调度策略的请求完成时间CDF对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # 保存图表
    plt.savefig('cdf.pdf',bbox_inches='tight')
    plt.close()

    
# 每种调度器执行1000个请求，然后画出各自的完成时间cdf图
def cdf(seed = 0, trait=True):
    if trait:
        results = {}
        trace_len = 2000
        for sched, info in scheduler.items():
            schedulerCls = scheduler_mapping[sched](**info)
            info = one_experiment(env=env, scheduler=schedulerCls, seed=seed, options={'trace_len': trace_len})
            results[sched] = list(info["all_request_process_time"].values())
            print(f"scheduler: {sched}")
            print(f"trace_len: {trace_len}")
            view = results[sched][:10]
            print(f"all_request_process_time: {view}...")
            print()

        # 保存为JSON文件
        with open('__result__/cdf.json', 'w') as f:
            json.dump(results, f, indent=4)
    else:
        with open('__result__/cdf.json', 'r') as f:
            results = json.load(f)

    # pprint(results)
    plot_cdf(results)

from collections import defaultdict
def all_metric_pic(seed = 0, trait = True):
    # request_len_array = [100,200,400,600,800,1000,1500,2000]
    request_len_array = [500,1000,1500,2000,2500,3000,3500,4000]
    if trait:
        results = defaultdict(lambda: defaultdict(list))

        # 增量更新
        # with open('__result__/all_metric.json', 'r') as f:
        #     results = json.load(f)
        #     for k in results:
        #         results[k]["ppo"] = []

        for sched, info in scheduler.items():
            schedulerCls = scheduler_mapping[sched](**info)
            for trace_len in request_len_array:
                info = one_experiment(env=env, scheduler=schedulerCls, seed=seed, options={'trace_len': trace_len})
                
                # 总处理时间
                results["total_request_process_time"][sched].append(sum(info["all_request_process_time"].values()))

                # 总执行时间
                results["all_task_execution_time"][sched].append(sum(info["all_task_execution_time"]))
                
                # 总下载时间
                results["total_download_time"][sched].append(sum(info["all_machine_download_time"]))

                # total request waiting time
                results["total_request_waiting_time"][sched].append(sum(info["all_task_waiting_time"]))

                # total down size
                results["total_download_size"][sched].append(sum(info["all_machine_download_size"]))

                # data tranmission time
                # results["total_data_tranmission_time"][sched].append(sum(info["all_machine_data_tranmission_time"]))

                results["total_request_wait_for_image"][sched].append(sum(info["all_task_wait_for_image"]))
                results["total_request_wait_for_data"][sched].append(sum(info["all_task_wait_for_data"]))
                results["total_request_wait_for_comp"][sched].append(sum(info["all_task_wait_for_comp"]))

                # pending download time
                # results["pending_download_time"][sched].append(?)
                
                print(f"scheduler: {sched}")
                print(f"trace_len: {trace_len}")
                print(f"total_request_process_time: {results['total_request_process_time'][sched][-1]}")
                print(f"total_download_time: {results['total_download_time'][sched][-1]}")
                # print(f"pending_download_time: {results['pending_download_time'][sched][-1]}")
                print(f"total_request_waiting_time: {results['total_request_waiting_time'][sched][-1]}")
                print(f"total_download_size: {results['total_download_size'][sched][-1]}")
                # print(f"total_data_tranmission_time: {results['total_data_tranmission_time'][sched][-1]}")
                print()

        # pprint(results)
        # 保存为JSON文件
        with open('__result__/all_metric.json', 'w') as f:
            json.dump(results, f, indent=4)

    else:
        with open('__result__/all_metric.json', 'r') as f:
            results = json.load(f)
            results


    plot_results(results["total_request_process_time"], request_len_array, y_label="总处理时间", fig_name="total_request_process_time", legend_pos="center left")

    plot_results(results["all_task_execution_time"], request_len_array, y_label="总计算时间", fig_name="all_task_execution_time")

    plot_results(results["total_download_time"], request_len_array,  y_label="总下载时间", fig_name="total_download_time")
    
    plot_results(results["total_request_waiting_time"], request_len_array, y_label="总等待时间", fig_name="total_request_waiting_time", legend_pos="center left")
    
    plot_results(results["total_download_size"], request_len_array, y_label="总下载大小", fig_name="total_download_size")

    # plot_results(results["total_data_tranmission_time"], request_len_array, "总传输时间对比", "total_data_tranmission_time")

    plot_results(results["total_request_wait_for_image"], request_len_array, y_label="总等待镜像时间", fig_name="total_request_wait_for_image",algos = ["PPO","DQN", "Dep-Wait", "Dep-Eft"])
    plot_results(results["total_request_wait_for_data"], request_len_array, y_label="总等待数据时间", fig_name="total_request_wait_for_data",algos = ["PPO","DQN", "Dep-Wait", "Dep-Eft"])
    plot_results(results["total_request_wait_for_comp"], request_len_array, y_label="总等待计算时间", fig_name="total_request_wait_for_comp",algos = ["PPO","DQN", "Dep-Wait", "Dep-Eft"])

    # plot_results(results["pending_download_time"], request_len_array, "总等待下载时间对比")


def plot_machine_distribution(data: dict, title="机器分布"):
    """画出各算法在不同机器上的任务分布柱状图"""
    
    # 1. 准备数据
    algorithms = list(data.keys())
    num_machines = len(list(data.values())[0])  # 机器数量
    x = np.arange(num_machines)  # 柱状图的x坐标
    width = 0.15  # 柱子的宽度
    
    # 2. 创建图表
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 3. 画出每个算法的柱子
    for idx, (algo, values) in enumerate(data.items()):
        ax.bar(x + idx*width, values, width, label=algo, color=colors.get(algo, '#1f77b4'))
    
    # 4. 设置图表属性
    ax.set_ylabel('函数数量')
    ax.set_xlabel('机器编号')
    # ax.set_title(title)
    ax.set_xticks(x + 2.5*width)
    ax.set_xticklabels([f'{i+1}' for i in range(num_machines)])
    ax.legend()
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # 5. 保存图表
    plt.savefig('machine_distribution.pdf',bbox_inches='tight',)
    plt.close()


def machine_distribution(seed=0, trait=True):
    if trait:
        results = {}
        trace_len = 2000
        for sched, info in scheduler.items():
            schedulerCls = scheduler_mapping[sched](**info)
            info = one_experiment(env=env, scheduler=schedulerCls, seed=seed, options={'trace_len': trace_len})
            results[sched] = info["machine_execute_task_num"]
            print(f"scheduler: {sched}")
            print(f"trace_len: {trace_len}")
            print(f"machine_execute_task_num: {results[sched]}...")
            print()

        # 保存为JSON文件
        with open('__result__/machine_distribution.json', 'w') as f:
            json.dump(results, f, indent=4)

    else:
        with open('__result__/machine_distribution.json', 'r') as f:
            results = json.load(f)

    # pprint(results)
    plot_machine_distribution(results)


# Tensorboard的平滑算法
def smooth(scalars, weight):  # weight是平滑因子
    last = scalars[0]
    smoothed = []
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed

def loss_pic():
    
    ppo_df = pd.read_csv("__result__/ppo_loss.csv")
    dqn_df = pd.read_csv("__result__/dqn_loss.csv")

    # 过滤数据
    ppo_df = ppo_df[ppo_df['Value'] > -150000]
    dqn_df = dqn_df[dqn_df['Value'] > -150000]
    
    # 3. 画PPO曲线
    plt.plot(ppo_df['Step'], ppo_df['Value'], 
            color='#ff7f0e',
            label='PPO',
            linewidth=2)
   
    
    # 4. 画DQN曲线
    plt.plot(dqn_df['Step'], dqn_df['Value'],
            color='#1f77b4',
            label='DQN',
            linewidth=2)
    
    # 5. 设置图表属性
    plt.xlabel('采样步数')
    plt.ylabel('奖励')
    # plt.title('训练过程中的奖励变化')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(loc='best')
    
    plt.savefig('loss.pdf', dpi=300, bbox_inches='tight')
    plt.close()

def plot_storage_comparison(results: dict, x_values: list, sched: str):
    plt.figure(figsize=(10, 6))
    
    # 为不同存储策略设置颜色和标记
    storage_colors = {
        'FCFS': '#1f77b4',     # 蓝色
        'LRU': '#ff7f0e',      # 橙色
        'Popularity': '#2ca02c',# 绿色
        'Priority': '#d62728',   # 红色
        # 'prioritySmall': '#8c564b'   # 红色
    }
    
    storage_markers = {
        'FCFS': 'o',      # 圆形
        'LRU': 's',       # 方形
        'Popularity': '^', # 三角形
        'Priority': 'D',   # 菱形
        # 'prioritysmall': 'p'   # 菱形
    }

    linestyles = {
        'Priority': 'solid',                    # 实线
        'Popularity': (0, (5, 1)),      # 长虚线         
        'LRU': (0, (3, 1, 1, 1)),      # 短线-点-点 
        'FCFS': 'dotted',              #点线
    }
    
    # 画出每个存储策略的曲线
    for storage_name, values in results.items():
        plt.plot(x_values, values,
                marker=storage_markers.get(storage_name, 'o'),
                color=storage_colors.get(storage_name, '#1f77b4'),
                linestyle=linestyles.get(storage_name, 'solid'),
                label=storage_name,
                linewidth=3,
                markersize=8)
    
    plt.xlabel('请求数量')
    plt.ylabel('总处理时间')
    # plt.title('不同存储策略的性能对比')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(f'{sched}-storage_comparison.pdf', bbox_inches='tight')
    plt.close()


def different_expel_strategy_test(seed=0, trait = False, sched = "dep-eft", schedulerMapping = None):
    storageMappling = {"FCFS": FCFSStorage, "LRU":LRUStorage
                  , "Popularity": PopularityStorage
                  , "Priority": PriorityStorage}
    

    request_len_array = [500,1000,1500,2000,2500,3000,3500,4000]

    if trait:
        results = defaultdict(lambda: defaultdict(list))
        for name, schedulerCls in schedulerMapping.items():
            storageCls = storageMappling[name]
            for trace_len in request_len_array:
                env = sim.LayerEdgeDynamicEnv(need_log=True, storage_type=storageCls)
                info = one_experiment(env=env, scheduler=schedulerCls, seed=seed, options={'trace_len': trace_len})

                results["total_request_process_time"][name].append(sum(info["all_request_process_time"].values()))

                print(f"storage: {name}")
                print(f"sched: {sched}")
                print(f"trace_len: {trace_len}")
                print(f"total_request_process_time: {results['total_request_process_time'][name][-1]}")
                print()

                # 保存为JSON文件
                with open(f'__result__/{sched}-different_expel_strategy.json', 'w') as f:
                    json.dump(results, f, indent=4)

    else:
        with open(f'__result__/{sched}-different_expel_strategy.json', 'r') as f:
            results = json.load(f)

    # print(results)
    plot_storage_comparison(results["total_request_process_time"], request_len_array, sched=sched)

def different_expel_strategy_all_test(seed=0, trait=True):
    scheds = {
        # "DQN": {
        #     "FCFS": scheduler_mapping["DQN"](config_path="config/dqn.yaml"),
        #     "LRU": scheduler_mapping["DQN"](config_path="config/dqn.yaml"),
        #     "Popularity": scheduler_mapping["DQN"](config_path="config/dqn.yaml"),
        #     "Priority": scheduler_mapping["DQN"](config_path="config/dqn.yaml")
        # },
        # "PPO": {
        #     "FCFS": scheduler_mapping["PPO"](config_path="config/ppo.yaml"),
        #     "LRU": scheduler_mapping["PPO"](config_path="config/ppo.yaml"),
        #     "Popularity": scheduler_mapping["PPO"](config_path="config/ppo.yaml"),
        #     "Priority": scheduler_mapping["PPO"](config_path="config/ppo.yaml")
        # },
        # "Dep-Eft": {
        #         "FCFS": scheduler_mapping["Dep-Eft"](**scheduler["Dep-Eft"]),
        #         "LRU": scheduler_mapping["Dep-Eft"](**scheduler["Dep-Eft"]),
        #         "Popularity": scheduler_mapping["Dep-Eft"](**scheduler["Dep-Eft"]),
        #         "Priority": scheduler_mapping["Dep-Eft"](**scheduler["Dep-Eft"]),
        # },
        # "Dep-Wait": {
        #     "FCFS": scheduler_mapping["Dep-Wait"](**scheduler["Dep-Wait"]),
        #     "LRU": scheduler_mapping["Dep-Wait"](**scheduler["Dep-Wait"]),
        #     "Popularity": scheduler_mapping["Dep-Wait"](**scheduler["Dep-Wait"]),
        #     "Priority": scheduler_mapping["Dep-Wait"](**scheduler["Dep-Wait"]),
        # },
        # "PPO": {
        #     "FCFS": scheduler_mapping["PPO"](config_path="config/ppo-fcfs.yaml"),
        #     "LRU": scheduler_mapping["PPO"](config_path="config/ppo-lru.yaml"),
        #     "Popularity": scheduler_mapping["PPO"](config_path="config/ppo-popularity.yaml"),
        #     "Priority": scheduler_mapping["PPO"](config_path="config/ppo.yaml")
        # },
        "DQN": {
            "FCFS": scheduler_mapping["DQN"](config_path="config/dqn-fcfs.yaml"),
            "LRU": scheduler_mapping["DQN"](config_path="config/dqn-lru.yaml"),
            "Popularity": scheduler_mapping["DQN"](config_path="config/dqn-popularity.yaml"),
            "Priority": scheduler_mapping["DQN"](config_path="config/dqn.yaml")
        },
    }

    for sched,schedulerMapping in scheds.items():
        different_expel_strategy_test(seed=seed, trait = trait, sched=sched, schedulerMapping=schedulerMapping)


def plot_predeploy_comparison(results: dict, x_values: list, sched: str):
    plt.figure(figsize=(10, 6))
    
    # 为不同存储策略设置颜色和标记
    colors = [
        '#1f77b4',     # 蓝色
        '#ff7f0e',     # 橙色
        '#d62728',      # 红色
        '#2ca02c',     # 绿色
    ]
    
    markers = [
        'o',      # 圆形
        's',       # 方形
        'D',   # 菱形
        '^', # 三角形
    ]

    linestyles = [
        (0, (5, 1)),        # 长虚线         
        (0, (3, 1, 1, 1)),  # 短线-点-点 
        'solid',            # 实线
        'dotted',           # 点线
    ]
    
    
    # 画出每个存储策略的曲线
    for i, (name, values) in enumerate(results.items()):
        plt.plot(x_values, values,
                marker=markers[i],
                color=colors[i],
                linestyle=linestyles[i],
                label=name,
                linewidth=3,
                markersize=8)
    
    plt.xlabel('请求数量')
    plt.ylabel('总处理时间')
    # plt.title("是否预部署对性能影响")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    plt.savefig(f'{sched}-predeploy_comparison.pdf', bbox_inches='tight')
    plt.close()

def predeploy_test(seed=0, trait = False):
    
    request_len_array = [500,1000,1500,2000,2500,3000,3500,4000]
    # request_len_array = [100,200,300,400,500,600,700,800]
    
    # sched = "Dep-Eft"
    # schedulerCls = scheduler_mapping[sched](**scheduler[sched])
    # envs = {
    #     sched+"-0": (sim.LayerEdgeDynamicEnv(need_log=True, is_predeploy=False), schedulerCls),

    #     sched+"-1": (sim.LayerEdgeDynamicEnv(need_log=True, is_predeploy=True, predeploy_degree=1), schedulerCls),
        
    #     sched+"-2": (sim.LayerEdgeDynamicEnv(need_log=True, is_predeploy=True, predeploy_degree=2), schedulerCls),

    #     sched+"-3": (sim.LayerEdgeDynamicEnv(need_log=True, is_predeploy=True, predeploy_degree=3), schedulerCls)
    # }

    sched = "PPO"
    ppo_origin = scheduler_mapping[sched](config_path="config/ppo.yaml")
    ppo_trained = scheduler_mapping["PPO"](config_path="config/ppo-predeploy-1.yaml")
    envs = {
        sched+"-0": (sim.LayerEdgeDynamicEnv(need_log=True, is_predeploy=False), ppo_origin),

        sched+"-1": (sim.LayerEdgeDynamicEnv(need_log=True, is_predeploy=True, predeploy_degree=1), ppo_origin),

        sched+"(trained)-1": (sim.LayerEdgeDynamicEnv(need_log=True, is_predeploy=True, predeploy_degree=1), ppo_trained),
    }

    if trait:
        results = defaultdict(lambda: defaultdict(list))
        # with open(f'__result__/{sched}-predeploy.json', 'r') as f:
        #     results = json.load(f)

        # for k in results:
        #     results[k]["PPO(trained)-1"] = []

        for name, (env, schedulerCls) in envs.items():
            for trace_len in request_len_array:
                info = one_experiment(env=env, scheduler=schedulerCls, seed=seed, options={'trace_len': trace_len})

                results["total_request_process_time"][name].append(sum(info["all_request_process_time"].values()))

                print(f"env: {name}")
                print(f"trace_len: {trace_len}")
                print(f"total_request_process_time: {results['total_request_process_time'][name][-1]}")
                print()

        # 保存为JSON文件
        with open(f'__result__/{sched}-predeploy.json', 'w') as f:
            json.dump(results, f, indent=4)

    else:
        with open(f'__result__/{sched}-predeploy.json', 'r') as f:
            results = json.load(f)

    # print(results)
    plot_predeploy_comparison(results["total_request_process_time"], request_len_array, sched = sched)

def difference():
    with open('__result__/all_metric.json', 'r') as f:
        results = json.load(f)
    # 获取两组指标数据
    wait_image = results["total_request_wait_for_image"]
    wait_data = results["total_request_wait_for_data"]
    
    differences = {}
    for algo in wait_image.keys():
        # 计算每个算法两组数据的相对差异
        image_data = np.array(wait_image[algo])
        data_data = np.array(wait_data[algo])
        
        # 使用相对差异：|a-b|/max(a,b)
        relative_diff = np.mean(
            np.abs(image_data - data_data) / image_data
        )
        
        differences[algo] = relative_diff
    
    # 按差异度排序
    sorted_algos = sorted(differences.items(), key=lambda x: x[1])
    
    print("算法差异度排序(越小表示两组数据差异越小):")
    for algo, diff in sorted_algos:
        print(f"{algo}: {diff:.4f}")

if __name__ == "__main__":
    # comparation()
    # test0()
    # xanadu_different_predeploy_degree()

    all_metric_pic(trait=False)
    # cdf(trait=False)
    # machine_distribution(trait=False)
    # loss_pic()

    # different_expel_strategy_all_test(trait=False)

    # predeploy_test(trait=False)

    # difference() 