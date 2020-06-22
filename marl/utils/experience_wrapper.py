import gym

class Experience_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super(Experience_Wrapper, self).__init__(env)
        self._data = []

    def reset(self, **kwargs):
        self._data = [(
            self.env.reset(**kwargs),
            None, None, None, None)]
        return self._data[0][0]

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self._data[-1][1:4] = action, r, done, info
        self._data.append((obs, None, None, None, None))
        return obs, r, done, info

    @property
    def data(self):
        return self._data