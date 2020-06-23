#bugs can belong to differing colonies, have different prey, and 
from . import Agent
from ..envs import Grid_World

class Moving_Agent(Agent):

    def __init__(self, env_type=Grid_World):
        """creates a 
        """
        self.env_type = env_type
        assert issubclass(env_type, Grid_World)
         
    def egocentric_obs(self, env): pass
    def egocentric_r(self, env): pass
    def egocentric_done(self, env): pass
    def egocentric_info(self, env): pass
    def apply_action(self, a, env):
        """attempts to apply an already
        decided action to environment"""
        pass
    
    @property
    def observation_space(self): pass
    @property
    def action_space(self): pass