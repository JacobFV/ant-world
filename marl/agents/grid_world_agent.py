#bugs can belong to differing colonies, have different prey, and 
from . import Agent
from ..envs.grid_world import Grid_World, Moving_Object, Signalling_Object

import numpy as np
import tensorflow as tf

FORWARD_SPEED_INDEX = 0
TURN_LEFT_INDEX = 1
TURN_RIGHT_INDEX = 2
PICK_INDEX = 3
PLACE_INDEX = 4
SIGNAL_INDEX_START = 5 # NOTE: do not change these signal
SIGNAL_INDEX_END = 7 # indecies without refactoring below

class Grid_World_Agent(Agent, Moving_Object, Signalling_Object):


    def __init__(self,
        initial_loc=None,
        initial_orientation=0,
        max_forward_speed=1.0,
        death_duration=10,
        **kwargs):
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
            loc=initial_loc, signal_depth=3, **kwargs)

        self.max_forward_speed = max_forward_speed
        self.orientation = initial_orientation
        self.alive = True
        self.death_duration = death_duration
         
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
        # decrease dramatically if not self.alive 
        pass
    def egocentric_done(self, env):
        return not self.alive

    def egocentric_info(self, env):
        return {}

    def apply_action(self, a, env):
        """attempts to apply an already
        decided action to environment"""
        # check if agent is alive
        if not self.alive:
            # poor agent can only die
            self.steps_dead += 1
            if self.steps_dead > self.death_duration:
                env.remove_agent(self)
            return

        if isinstance(a, tf.Variable):
            a = a.numpy()

        # try to move
        delta_loc = self.max_forward_speed * 
            a[FORWARD_SPEED_INDEX] * [
                round(np.cos(orientation)),
                round(np.sin(orientation)),
            ]
        self.try_move(delta_loc, env)

        # try rotating
        if a[TURN_LEFT_INDEX] > 0.5:
            self.orientation += np.pi/2
        if a[TURN_RIGHT_INDEX] > 0.5:
            self.orientation -= np.pi/2

        # make signals
        self._set_signal(a[SIGNAL_INDEX_START:SIGNAL_INDEX_END])

        raise NotImplementedError()

    @property
    def observation_space(self):
        return (4+0,) #TODO
    @property
    def action_space(self):
        return (4,) #TODO

    def die(self, env):
        """in the apply_action method
        another agent may call `die` on self
        This will make it begin dying until
        it dissapears from the environment"""
        self.alive = False
        self.steps_dead = 0