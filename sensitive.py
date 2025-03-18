
from data.config import Config
from data.data_generate import DataGenerator
import sim
from schedulers import scheduler_mapping, Scheduler
from main import one_experiment,scheduler,plot_results
from collections import defaultdict
import json

_func_comp = [0.5, 1, 1.5, 2, 2.5, 3] # 函数请求的平均计算资源

_d = [0.1, 0.5, 1, 1.5, 2] # 函数之间传输数据的平均大小

_edge_delay = [0.1, 0.5, 1, 1.5, 2]

_gamma = [0.1, 0.5, 1, 1.5, 2]

num_containers = [100, 300, 500, 700, 900]

num_layers = [400, 700, 1000, 1300, 1600]

trace_len = 2000
seed = 0
scheduler_name = ["Dep-Eft", "Dep-Wait", "DQN", "PPO"]
# scheduler_name = ["dqn"]
# scheduler: Scheduler = scheduler_mapping["ppo"](config_path="config/ppo.yaml")


def sensitive_experiment(sensitive_params
             , param_name
             , human_name
             , trait=False):
    if trait:
        results = defaultdict(lambda: defaultdict(list))
        # 增量更新
        # with open(f'__result__/sensitive_{param_name}.json', 'r') as f:
        #     results = json.load(f)
        #     for k in results:
        #         results[k]["dqn"] = []

        for c in sensitive_params:
            print(f"{param_name}: {c}")
            config = Config()
            config[param_name] = c
            # config._func_comp = c
            workload_data = f'data/workload_data_{param_name}_{c}'
            DataGenerator().generate(config).save(workload_data)

            env = sim.LayerEdgeDynamicEnv(need_log=True, workload_data=workload_data)

            for name in scheduler_name:
                schedulerCls = scheduler_mapping[name](**scheduler[name])
                infos = one_experiment(env=env, scheduler=schedulerCls, seed=seed, options={'trace_len': trace_len})

                print("sched: ", name)
                # results[name].append(sum(infos["all_request_process_time"].values()))

                # 总处理时间
                results["total_request_process_time"][name].append(sum(infos["all_request_process_time"].values()))

                # 总执行时间
                results["all_task_execution_time"][name].append(sum(infos["all_task_execution_time"]))
                
                # 总下载时间
                results["total_download_time"][name].append(sum(infos["all_machine_download_time"]))

                # total request waiting time
                results["total_request_waiting_time"][name].append(sum(infos["all_task_waiting_time"]))

                # total down size
                results["total_download_size"][name].append(sum(infos["all_machine_download_size"]))

                # data tranmission time
                # results["total_data_tranmission_time"][sched].append(sum(info["all_machine_data_tranmission_time"]))

                results["total_request_wait_for_image"][name].append(sum(infos["all_task_wait_for_image"]))
                results["total_request_wait_for_data"][name].append(sum(infos["all_task_wait_for_data"]))
                results["total_request_wait_for_comp"][name].append(sum(infos["all_task_wait_for_comp"]))

                print(infos["machine_execute_task_num"])

                print(f"total_request_process_time: {results['total_request_process_time'][name][-1]}")
                print(f"total_request_waiting_time: {results['total_request_waiting_time'][name][-1]}")
                print(f"total_request_wait_for_image: {results['total_request_wait_for_image'][name][-1]}")
                print(f"total_request_wait_for_data: {results['total_request_wait_for_data'][name][-1]}")
                print(f"total_request_wait_for_comp: {results['total_request_wait_for_comp'][name][-1]}")
                print()
        # 保存为JSON文件
        with open(f'__result__/sensitive_{param_name}.json', 'w') as f:
            json.dump(results, f, indent=4)
    else:
        with open(f'__result__/sensitive_{param_name}.json', 'r') as f:
            results = json.load(f)

    plot_results(results["total_request_process_time"], sensitive_params, x_label=human_name, fig_name=f"sensitive_{param_name}", algos=scheduler_name, threshold=150000)

    # plot_results(results["total_request_waiting_time"], sensitive_params, x_label=human_name, fig_name=f"sensitive_{param_name}_wait", algos=scheduler_name)

    # plot_results(results["total_request_wait_for_image"], sensitive_params, x_label=human_name, fig_name=f"sensitive_{param_name}_wait_for_image", algos=scheduler_name)

    # plot_results(results["total_request_wait_for_data"], sensitive_params, x_label=human_name, fig_name=f"sensitive_{param_name}_wait_for_data", algos=scheduler_name)

    # plot_results(results["total_request_wait_for_comp"], sensitive_params, x_label=human_name, fig_name=f"sensitive_{param_name}_wait_for_comp", algos=scheduler_name)




sensitive_experiment(sensitive_params=_func_comp
                        , param_name="_func_comp"
                        , human_name="平均计算消耗"
                        , trait=True)

sensitive_experiment(sensitive_params=_d
                        , param_name="_d"
                        , human_name="平均数据传输大小"
                        , trait=True)

sensitive_experiment(sensitive_params=_edge_delay
                        , param_name="_edge_delay"
                        , human_name="平均数据传输延迟"
                        , trait=True)

sensitive_experiment(sensitive_params=_gamma
                        , param_name="_gamma"
                        , human_name="平均镜像拉取延迟"
                        , trait=True)

sensitive_experiment(sensitive_params=num_containers
                        , param_name="num_containers"
                        , human_name="容器数量"
                        , trait=True)

sensitive_experiment(sensitive_params=num_layers
                        , param_name="num_layers"
                        , human_name="镜像层数量"
                         , trait=True)


