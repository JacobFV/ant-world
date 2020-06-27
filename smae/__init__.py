from .env import SMAE

# register social multiagent environment in gym
import gym
gym.envs.register(
    id='smae',
    entry_point='smae.env:smae',
    max_episode_steps=-1,
    kwargs = {
        'world_size': (16, 16, 1)
        #TODO
    }
)