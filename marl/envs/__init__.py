import gym

class MA_Gym_Env(gym.Env):

    def __init__(self, agents):
        """
        args:
            agents: array of agents to inhabit the world

        NOTE: instead of initializing the env and
        only later the agents with their obs/action
        specs, agents must have a predefined spec
        and already be initialized
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

class Grid_World(MA_Gym_Env):

    def __init__(self, world_size=(16,16,1), **kwargs):
        """
        args:
            world_size: n-tuple of world size (can have dim > 2)
        """
        super(Grid_World, self).__init__(**kwargs)
        self.world_size = world_size

    def _global_update(self, a_n):
        #TODO actually there may be no global update code needed
        pass

    def render(self, mode="human"):
        """renders entire environment as bitmap"""
        #TODO
        pass