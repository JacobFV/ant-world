#bugs can belong to differing colonies, have different prey, and 
from . import Agent
from ..envs.grid_world import Grid_World, Moving_Object

import numpy as np
import tensorflow as tf

FORWARD_SPEED_INDEX = 0
TURN_LEFT_INDEX = 1
TURN_RIGHT_INDEX = 2
PICK_INDEX = 3
PLACE_INDEX = 4
SIGNAL_INDECIES = (5, 6, 7)

class Grid_World_Agent(Agent, Moving_Object):


    def __init__(self, initial_loc=None, initial_orientation=0, max_forward_speed=1.0, **kwargs):
        """creates a moving agent
        works only in Grid_World

        args:
            initial_loc: 
            initial_orientation: radian angle (-inf, inf)
                on the plane formed by first two dimensions
            max_forward_speed: maximum speed moving forward
            algorithm: see Agent
            ops: np.int8. see Moving_Object
        """
        if initial_loc is None:
            raise NotImplementedError()
        if initial_orientation is None:
            raise NotImplementedError()

        super(Grid_World_Agent, self).__init__(
            loc=initial_loc, **kwargs)

        self.max_forward_speed = max_forward_speed
        self.orientation = [
            round(np.cos(initial_orientation)),
            round(np.sin(initial_orientation)),
        ]
         
    def egocentric_obs(self, env):
        # {
        #    "allowed_ops",
        #    "moving_obj_data",
        #    "my signal" (stateful from self)
        # }
        pass
    def egocentric_r(self, env):
        # TODO should obs, a be passed in here?
        # increase if actions included `pick` edible block
        # decrease reward with energy spent moving
        # decrease with energy spent sharing
        pass
    def egocentric_done(self, env): pass
    def egocentric_info(self, env): pass

    def apply_action(self, a, env):
        """attempts to apply an already
        decided action to environment"""
        if isinstance(a, tf.Variable):
            a = a.numpy()

        # try move
        delta_loc = self.orientation * \
            self.max_forward_speed * 
            a[FORWARD_SPEED_INDEX]
        self.try_move(delta_loc, env)

        # try rotate
        if a[TURN_LEFT_INDEX] > 0.5:
        pass
    
    @property
    def observation_space(self):
        return (4+0,)
    @property
    def action_space(self):
        return (4,)