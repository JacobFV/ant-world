import gym

class Experience_Wrapper(gym.Wrapper):
    def __init__(self, env):
        super(Experience_Wrapper, self).__init__(env)
        self._episode = []

    def reset(self, **kwargs):
        self._episode = [(
            self.env.reset(**kwargs),
            None, None, None, None)]
        return self._episode[0][0]

    def step(self, action):
        obs, r, done, info = self.env.step(action)
        self._episode[-1][1:4] = action, r, done, info
        self._episode.append((obs, None, None, None, None))
        return obs, r, done, info

    @property
    def episode(self):
        return self._episode
        
def multiagent_train(episode):
    """
    Train a multi-agent rl system on its record.
    This algorithm works with varying amounts
    of agents at different time steps.

    args:
        episode: list of (obs_n, a_n,
            r_n, done_n, info_n) tuples

    return: returns nothing"""

    # identify unique agents
    unique_agents = []
    for obs_n, a_n, r_n, done_n, info_n in episode:
        for agent in obs_n.keys():
            if agent not in unique_agents:
                unique_agents.append(agent)

    # optimize over entire episode one agent at a time
    for agent in unique_agents:
        # generate agent-centric trajectories and train on them
        agent_episode = []
        for obs_n, a_n, r_n, done_n, info_n in episode:
            # some agents only exist in some frames
            if agent not in obs_n.keys():
                continue
            agent_episode.append(
                obs_n[agent],
                a_n[agent],
                r_n[agent],
                done_n[agent],
                info_n[agent]
            )
        
        # optimize over entire episode at once
        agent.train(agent_episode)