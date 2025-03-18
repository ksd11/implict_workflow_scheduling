from abc import ABC, abstractmethod
from gymnasium import Wrapper

import torch.nn as nn


class Scheduler(ABC):
    """Interface for all schedulers"""

    name: str
    env_wrapper_cls: type[Wrapper]

    @abstractmethod
    def schedule(self, obs: list) -> tuple[int, dict]:
        pass