from .LayerEdgeEnv import LayerEdgeEnv
import gymnasium as gym

gym.register(
    id='LayerEdgeEnv-v0',
    entry_point='sim.LayerEdgeEnv:LayerEdgeEnv'
)