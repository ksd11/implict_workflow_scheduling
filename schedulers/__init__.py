from .scheduler import Scheduler
from .trainable import TrainableScheduler
from .random import RandomScheduler
from .xanadu import XanaduScheduler
from .dep_down import DepDownScheduler
from .dep_wait import DepWaitScheduler
from .dep_eft import DepEFTScheduler
from .trainable_predeploy import TrainablePredeployScheduler

scheduler_mapping = {
    "dqn": TrainableScheduler,
    "ppo": TrainableScheduler,
    "random": RandomScheduler,
    "xanadu": XanaduScheduler,
    "dep-down": DepDownScheduler,
    "dep-wait": DepWaitScheduler,
    "dep-eft": DepEFTScheduler,
    "ppo-predeploy": TrainablePredeployScheduler
}