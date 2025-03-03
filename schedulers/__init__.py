from .scheduler import Scheduler
from .greedy import GreedyScheduler
from .trainable import TrainableScheduler
from .random import RandomScheduler

scheduler_mapping = {
    "dqn": TrainableScheduler,
    "ppo": TrainableScheduler,
    "random": RandomScheduler,
    "greedy": GreedyScheduler
}