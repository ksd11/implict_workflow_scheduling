import gymnasium as gym


#定义环境
class MyWrapper(gym.Wrapper):
    metadata = {
        # "max_episode_steps": 200
        "render_modes": ["rgb_array","human"]
    }

    def __init__(self, render_mode="rgb_array"):
        env = gym.make('CliffWalking-v0', render_mode=render_mode)
        super().__init__(env)
        self.env = env
        self.step_n = 0

    def reset(self, seed=None, options=None):
        state, info = self.env.reset(seed=seed, options=options)
        self.step_n = 0
        return state, info

    def step(self, action):
        state, reward, done, truncated, info = self.env.step(action)
        self.step_n += 1
        if self.step_n >= 200:
            done = True
        return state, reward, done, truncated, info
