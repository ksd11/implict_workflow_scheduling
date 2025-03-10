import sim
from cfg_loader import parse
from trainers import make_trainer, play_a_game
import gymnasium as gym

args = parse()

def main():    
    if args["test_mode"]:
        args["cfg"]["env"]["render_mode"] = "human"
        print(play_a_game(args["cfg"]))
    else:
        make_trainer(args["cfg"], load=False).train()    

main()

# 性能分析
# 方法1: 保存到文件
# import cProfile
# # 创建性能分析器
# pr = cProfile.Profile()
# # 运行代码
# pr.enable()
# main()
# pr.disable()
# # 保存结果到文件
# pr.dump_stats('profile.stats')

# # 方法2: 直接打印结果，按不同指标排序
# # sort 选项:
# # - 'cumtime': 累计时间
# # - 'tottime': 总时间
# # - 'calls': 调用次数
# # - 'ncalls': 调用次数 
# pr.print_stats(sort='cumtime')

# 方法4: 按正则表达式过滤
# pr.print_stats('foo:')  # 只显示包含'foo'的函数
