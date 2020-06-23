#bugs can belong to differing colonies, have different prey, and 
from . import Agent
class Bug(Agent):

    def __init__(self, env_type):
        self.env_type = env_type
         
    def egocentric_obs(self, env): pass
    def egocentric_r(self, env): pass
    def egocentric_done(self, env): pass
    def egocentric_info(self, env): pass
    def apply_action(self, a, env):
        """attempts to apply an already
        decided action to environment"""
        pass