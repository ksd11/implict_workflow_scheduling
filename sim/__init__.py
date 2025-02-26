from .LayerEdgeEnv import LayerEdgeEnv
from .wrapper import MyWrapper
import gymnasium as gym

gym.register(
    id='LayerEdgeEnv-v0',
    entry_point='sim.LayerEdgeEnv:LayerEdgeEnv'
)

gym.register(
    id='MyWrapper-v0',
    entry_point='sim.wrapper:MyWrapper'
)