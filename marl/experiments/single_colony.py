from ..worlds import Grid_World
from ..agents.bug import Bug
from ..agents.algorithms.model_based import Model_Based_Algorithm

import gym

agents = [Bug(Model_Based_Algorithm)] * 10

env = Grid_World(world_size=(16,16), agents=agents)
env = gym.wrappers.Monitor(env, "recording")

obs_n = env.reset()
for t in range(100):
    a_n = {agent: agent.act(obs)
            for agent, obs in obs_n.items()}
    obs_n, r_n, done_n, info_n = env.step(a_n)