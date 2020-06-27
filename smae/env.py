import numpy as np
import tensorflow as tf
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
        # since agents may die and be added to self.agents
        # mid step, a frozen copy is used for this step
        origonal_agents = self.agents.copy()
        for agent, a in a_n.items():
            agent.apply_action(a, self)
        self._global_update(a_n)
        return {
            agent: agent.egocentric_obs(self)
            for agent in origonal_agents
        }, {
            agent: agent.egocentric_r(self)
            for agent in origonal_agents
        }, {
            agent: agent.egocentric_done(self)
            for agent in origonal_agents
        }, {
            agent: agent.egocentric_info(self)
            for agent in origonal_agents
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


class Social_MA_Env(MA_Gym_Env):

    metadata = {'render.modes': ["rgb"]}
    
    def __init__(self, world_size=(16,16,4), static_objects=None, gravity=(0,0,-1), **kwargs):
        """
        args:
            world_size: n-tuple of world size (can have dim > 2)
            static_objects: numpy int8 array specifying allowed ops
        """
        if static_objects is None:
            static_objects = np.zeros(world_size, dtype=np.int8)

        super(Social_MA_Env, self).__init__(**kwargs)

        for agent in self.agents:
            assert isinstance(agent, Moving_Object)

        self.world_size = world_size
        self.gravity = gravity

        self.static_objects = static_objects
        self.moving_objects = self.agents
        self._update_combined_objects()

    def moving_object_at(self, loc):
        """returns the moving object (if present)
        at loc. returns `None` if just static objects"""
        for moving_object in self.moving_objects:
            if moving_object.loc == loc:
                return moving_object
        return None

    def render(self, mode="rgb"):
        """renders entire environment as bitmap"""
        #TODO
        pass

    def add_agent(self, agent):
        """new agents created midgame are
        added by their parents"""
        super(Grid_World, self).add_agent(agent)
        self.moving_objects.append(agent)

    def remove_agent(self, agent):
        """agents that die midgame call this method"""
        super(Grid_World, self).remove_agent(agent)
        self.moving_objects.remove(agent)

    def _global_update(self, a_n):
        self._update_combined_objects()
        self._apply_gravity()

    def _update_combined_objects(self):
        """update self.combined_objects with
        new moving_object locations"""
        self.combined_objects = self.static_objects.copy()
        for moving_object in self.moving_objects:
            self.combined_objects[moving_object.loc] = \
                moving_object.ops

    def _apply_gravity(self):
        for moving_object in self.moving_objects:
            moving_object.try_move(self.gravity, self)