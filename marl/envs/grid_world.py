from ..envs.base import MA_Gym_Env

class Grid_World(MA_Gym_Env):

    def __init__(self, world_size=(16,16,1), **kwargs):
        """
        args:
            world_size: n-tuple of world size (can have dim > 2)
        """
        super(Grid_World, self).__init__(**kwargs)
        self.world_size = world_size

    def _global_update(self, a_n):
        #there may be no global update code needed
        pass

    def render(self, mode="human"):
        """renders entire environment as bitmap"""
        #TODO
        pass