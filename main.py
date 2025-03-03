
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from cfg_loader import load
from pprint import pprint
import sim
from schedulers import scheduler_mapping, Scheduler

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

env = sim.LayerEdgeEnv()
# env = sim.LayerEdgeDynamicEnv()
scheduler = {
    "dqn":{
        "config_path": "config/dqn.yaml"
    },
    "ppo":{
        "config_path": "config/ppo.yaml"
    }, 
    "greedy": {
        "edge_server_num": env.N,
        "layer_num": env.L
    }, 
    "random": {
        "edge_server_num": env.N,
        "layer_num": env.L
    }}

def report(info:dict):
    # for k in info:
        # print(info[k])
    makespan = max([info[k]['finish_time'] for k in info])
    # print("Makespan is: ", makespan)
    return makespan

def one_experiment(env, scheduler: Scheduler, seed = None, options = {'trace_len': 100}):

    state,_ = env.reset(seed=seed, options=options)
    reward_sum = 0
    done = False

    while not done:
        action ,_state = scheduler.schedule(state)
        state, reward, terminated, truncated, info = env.step(int(action))
        done = terminated or truncated
        reward_sum += reward
        env.render()

    env.close()
    # pprint(info)
    makespan = report(info['schedule_info'])
    # return reward_sum
    return {
        "reward_sum": reward_sum,
        "makespan": makespan
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



if __name__ == "__main__":
    # params = make_parser().parse_args()
    # cfg = load(params.filename)
    # env = sim.LayerEdgeDynamicEnv()

    # scheduler = schedulers.GreedyScheduler(env.N, env.L)
    # scheduler = schedulers.TrainableScheduler(cfg)
    # scheduler = schedulers.RandomScheduler(env.N, env.L)

    results = {}
    request_len_array = [100,200,400,600,800,1000]
    for sched, info in scheduler.items():
        scheduler = scheduler_mapping[sched](**info)
        results[sched] = []
        for trace_len in request_len_array:
            info = one_experiment(env=env, scheduler=scheduler, seed=0, options={'trace_len': trace_len})
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
        "greedy": results["greedy"]
    }

    plot_results(results, request_len_array, "不同算法在不同请求数量下的完成时间对比")
