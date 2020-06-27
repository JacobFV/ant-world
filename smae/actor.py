from .elements import Moving_Object, Signalling_Object, OPERATIONS

import numpy as np
import tensorflow as tf
import gym

# observation space constants
OBS_OPERATIONS = "OPERATIONS"
OBS_SIGNALS = "SIGNALS"
OBS_MY_SIGNAL = "MY_SIGNAL"

# action space constants
ACT_FORWARD_SPEED_INDEX = 0
ACT_TURN_LEFT_INDEX = 1
ACT_TURN_RIGHT_INDEX = 2
ACT_PICK_INDEX = 3
ACT_PLACE_INDEX = 4
ACT_SIGNAL_INDEX_START = 5
ACT_SIGNAL_DEPTH = 4
ACT_SIGNAL_INDEX_END = SIGNAL_INDEX_START + SIGNAL_DEPTH
ACT_LENGTH = ACT_SIGNAL_INDEX_END

class Actor(Moving_Object, Signalling_Object):

    def __init__(self,
        env,
        initial_loc,
        initial_orientation=0.0,
        max_forward_speed=1.0,
        death_duration=10,
        vision_size=(5, 8, 1)):
        """creates a moving agent
        works only in Grid_World

        In egocentric coordinates,
            x is left and right
            y is forward and backward
            z is up and down

        args:
            initial_loc: 
            initial_orientation: radian angle (-inf, inf)
                on the plane formed by first two dimensions
            max_forward_speed: maximum speed moving forward
            algorithm: see Agent
            vision_size: egocentric vision boundries
        """

        super(Actor, self).__init__(
            loc=initial_loc,
            signal_depth=SIGNAL_DEPTH,
            ops=[OPERATIONS.PICKUP, OPERATIONS.PUSH_OVER])

        self.env = env
        self.max_forward_speed = max_forward_speed
        self.orientation = initial_orientation
        self.alive = True
        self.death_duration = death_duration
        self.vision_size = vision_size
         
    def egocentric_obs(self, env):
        # {
        #    "allowed_ops",
        #    "moving_obj_data",
        #    "my signal" (stateful from self)
        # }
        pass
    def egocentric_r(self, env, obs, a):
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

        assert a >= np.zeros_like(a)

        # try to move
        delta_loc = self.max_forward_speed * 
            a[ACT_FORWARD_SPEED_INDEX] * [
                np.cos(orientation),
                np.sin(orientation),
            ]
        self.try_move(delta_loc, env)

        # try rotating
        self.orientation += (np.pi/2) * (
            a[ACT_TURN_LEFT_INDEX] - a[ACT_TURN_RIGHT_INDEX])

        # make signals
        self._set_signal(
            a[ACT_SIGNAL_INDEX_START:ACT_SIGNAL_INDEX_END])

        return

    @property
    def observation_space(self):
        return gym.spaces.Dict({
            OBS_OPERATIONS: gym.spaces.Box(
                low=0,
                high=255,
                shape=self.vision_size,
                dtype=np.int8
            ),
            OBS_SIGNALS: gym.spaces.Box(
                low=0,
                high=1,
                shape=self.vision_size+(ACT_SIGNAL_DEPTH,),
                dtype=np.float
            ),
            OBS_MY_SIGNAL: gym.spaces.Box(
                low=0,
                high=1,
                shape=(ACT_SIGNAL_DEPTH,),
                dtype=np.float
            )
        })

    @property
    def action_space(self):
        return gym.spaces.Box(
            low=0,
            high=1,
            shape=(ACT_LENGTH, ),
            dtype=np.float
        )

    def die(self, env):
        """in the apply_action method
        another agent may call `die` on self
        This will make it begin dying until
        it dissapears from the environment"""
        self.alive = False
        self.steps_dead = 0