import gym

class MA_Gym_Env(gym.Env):

    def __init__(self, agents):
        """
        args:
            agents: array of agents to initially
                inhabit the world

        NOTE: agents can be initialized later or
            even during the simulation and added to
            the environment with `self.add_agent(agent)`
        """
        self.agents = agents

    def reset(self):
        """returns: vectorized observation for agents"""
        return {
            agent: agent.egocentric_obs(self)
            for agent in self.agents
        }

    def _global_update(self, a_n):
        """since the agents have already individually
        attempted to execute their actions, `a_n` may
        be disregaurded but is still passed along in
        case it serves any purpose later"""
        pass
    def step(self, a_n):
        """simulate one step
        args:
            action_n: vectorized actions for agents
        returns: tuple of vectorized agent data
                (obs_n, r_n, done_n, info_n)"""
        for agent, a in a_n.items():
            agent.apply_action(a, self)
        self._global_update(a_n)
        return {
            agent: agent.egocentric_obs(self)
            for agent in self.agents
        }, {
            agent: agent.egocentric_r(self)
            for agent in self.agents
        }, {
            agent: agent.egocentric_done(self)
            for agent in self.agents
        }, {
            agent: agent.egocentric_info(self)
            for agent in self.agents
        }

    def close(self):
        """free any resources not automatically destroyed"""
        pass
    def render(self, mode="human"):
        """renders entire environment"""
        pass

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            agent: agent.observation_space
            for agent in self.agents
        })

    @property
    def action_space(self):
        return gym.spaces.Dict({
            agent: agent.action_space
            for agent in self.agents
        })

    def add_agent(self, agent):
        """new agents created midgame are
        added by their parents"""
        self.agents.append(agent)

    def remove_agent(self, agent):
        """agents that die midgame call this method"""
        self.agents.remove(agent)