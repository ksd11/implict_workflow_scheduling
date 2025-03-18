from .scheduler import Scheduler
from .trainable import TrainableScheduler
from .random import RandomScheduler
from .xanadu import XanaduScheduler
from .dep_down import DepDownScheduler
from .dep_wait import DepWaitScheduler
from .dep_eft import DepEFTScheduler
from .trainable_predeploy import TrainablePredeployScheduler

scheduler_mapping = {
    "DQN": TrainableScheduler,
    "PPO": TrainableScheduler,
    "Random": RandomScheduler,
    "xanadu": XanaduScheduler,
    "Dep-Down": DepDownScheduler,
    "Dep-Wait": DepWaitScheduler,
    "Dep-Eft": DepEFTScheduler,
    "ppo-predeploy": TrainablePredeployScheduler
}