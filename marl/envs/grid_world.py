from ..envs.base import MA_Gym_Env
import numpy as np

class Moving_Object:
    def __init__(self, loc: list, ops: np.int8):
        self.loc = loc
        self.ops = ops
    
    def try_move(self, delta_loc, env):
        raise NotImplementedError() #TODO

class Grid_World(MA_Gym_Env):

    metadata = {'render.modes': ["rgb"]}
    
    def __init__(self, world_size=(16,16,4), static_objects=None, gravity=(0,0,-1), **kwargs):
        """
        args:
            world_size: n-tuple of world size (can have dim > 2)
            static_objects: numpy int8 array specifying allowed ops
        """
        if static_objects is None:
            static_objects = np.zeros(world_size, dtype=np.int8)

        super(Grid_World, self).__init__(**kwargs)

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