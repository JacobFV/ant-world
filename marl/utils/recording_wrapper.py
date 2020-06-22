import numpy as np
import gym

class Recording_Wrapper(gym.Wrapper):
    def __init__(self, env, skip_rate=16):
        super(Recording_Wrapper, self).__init__(env)
        self.frame = 0
        self.skip_rate = skip_rate
        self._frames = []

    def step(self, action):
        self.frame += 1
        if self.frame % self.skip_rate == 0:
            self._frames.append(self.env.render())
        return self.env.step(action)

    @property
    def frames(self):
        return self._frames

    def save_video(self, filepath):
        raise NotImplementedError("Sorry. Not yet implimented")