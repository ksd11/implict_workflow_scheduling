from .scheduler import Scheduler
from .greedy import GreedyScheduler
from .trainable import TrainableScheduler
from .random import RandomScheduler
from .xanadu import XanaduScheduler

scheduler_mapping = {
    "dqn": TrainableScheduler,
    "ppo": TrainableScheduler,
    "random": RandomScheduler,
    "greedy": GreedyScheduler,
    "xanadu": XanaduScheduler
}