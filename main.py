
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from cfg_loader import load
from pprint import pprint
import sim
import schedulers

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

def report(info:dict):
    for k in info:
        print(info[k])
    makespan = max([info[k]['finish_time'] for k in info])
    print("Makespan is: ", makespan)

def one_experiment(env, scheduler: schedulers.Scheduler):
    state,_ = env.reset()
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
    report(info['schedule_info'])
    return reward_sum

if __name__ == "__main__":
    params = make_parser().parse_args()
    cfg = load(params.filename)
    # env = sim.LayerEdgeEnv()
    env = sim.LayerEdgeDynamicEnv()

    # scheduler = schedulers.GreedyScheduler(env.N, env.L)
    # scheduler = schedulers.TrainableScheduler(cfg)
    scheduler = schedulers.RandomScheduler(env.N, env.L)

    one_experiment(env, scheduler)
