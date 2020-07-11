from .elements import Moving_Object, Signaling_Moving_Object, OPERATIONS

import numpy as np
import tensorflow as tf
import gym

# observation space constants
OBS_OPERATIONS = "OBS_OPERATIONS"
OBS_SIGNALS = "OBS_SIGNALS"
OBS_MY_SIGNAL = "OBS_MY_SIGNAL"
OBS_FREE_STORAGE_PERCENT = "OBS_FREE_STORAGE_PERCENT"
OBS_HEALTH = "OBS_HEALTH"
OBS_REWARD = "OBS_REWARD"

# action space constants
ACT_SIGNAL = "ACT_SIGNAL"
ACT_CONTINUOUS = "ACT_CONTINUOUS"
ACT_CONTINUOUS_LEN = 6

ACT_FORWARD_SPEED_INDEX = 0
ACT_TURN_LEFT_INDEX = 1
ACT_TURN_RIGHT_INDEX = 2
ACT_PICK_INDEX = 3
ACT_PLACE_INDEX = 4
ACT_EAT_INDEX = 5

VOCAB_SIZE = 1024

RESTING_ENERGY_RATE = 0.2 # energy consumed at every step
FOOD_ENERGY = 10 # energy gained by eating food
EATING_COST = 0.5
MOVING_ENERGY_COST = 1 # energy spent per 1 unit-mass moved
PICKUP_COST = 1 # energy spend picking up 1 mass
FAILED_PICKUP_COST = 1 # additional energy spent when picking up fails
ATTACKING_COST = 2 # energy spent attacking
ATTACKING_GAIN_COEF = 0.8 # coefficent of energy gained from attacking
PLACE_COST = 1 # energy spend placing 1 mass
FAILED_PLACE_COST = 1 # additional energy spent when olacing fails

class Actor(Signaling_Moving_Object):

    def __init__(self,
        env,
        initial_loc:tuple=None,
        initial_orientation:float=0.0,
        max_forward_speed:float=1.0,
        vision_size:tuple=(5, 8, 1),
        initial_energy=100.0,
        storage_capacity=3):
        """creates a moving agent
        works only in Grid_World

        In egocentric coordinates,
            x is left and right
            y is forward and backward
            z is up and down

        args:
            initial_loc: tuple. If None (default), location
                is randomly initialized by env.random_avaliable_loc
            initial_orientation: radian angle (-inf, inf)
                on the plane formed by first two dimensions
            max_forward_speed: maximum speed moving forward
            algorithm: see Agent
            vision_size: egocentric dimensionwise vision lengths.
                The second axis is not symetric; all other axes are.
                This gives the agent a 180deg field of view. E.G.:
                (3, 8, 0) allows agents to see 3 units left, 3 units
                right, (and the x location they are at), 8 units
                foreward (including their present location), and only
                in-plane with their current elevation
            initial_energy: energy the actor begins with (100 default)
            storage_capacity: amount of items that the actor can store
                at once (default 3)
        """
        initial_loc = env.random_avaliable_loc() \
            if initial_loc is None else initial_loc
        super(Actor, self).__init__(
            loc=initial_loc,
            signal_depth=VOCAB_SIZE,
            ops=[OPERATIONS.PICKUP,
                OPERATIONS.PUSH_OVER,
                OPERATIONS.EAT])

        self.env = env
        self.max_forward_speed = max_forward_speed
        self.orientation = initial_orientation
        self.vision_size = vision_size
        self.energy = initial_energy
        self.storage = []
        self.storage_capacity = storage_capacity
        self.prev_energy = self.energy
        self.reward = 0.0
         
    def egocentric_obs(self, env):
        # NOTE non idempotent logic here
        self._calc_energy_gain_reward()

        return {
            OBS_OPERATIONS: env.combined_object_ops[
                self.rounded_loc[0] - self.vision_size[0]:
                self.rounded_loc[0] + self.vision_size[0],
                self.rounded_loc[1]: # only 180deg FOV
                self.rounded_loc[1] + self.vision_size[1],
                self.rounded_loc[2] - self.vision_size[2]:
                self.rounded_loc[2] + self.vision_size[2],
            ],
            OBS_SIGNALS: env.signal_field[
                self.rounded_loc[0] - self.vision_size[0]:
                self.rounded_loc[0] + self.vision_size[0],
                self.rounded_loc[1]: # only 180deg FOV
                self.rounded_loc[1] + self.vision_size[1],
                self.rounded_loc[2] - self.vision_size[2]:
                self.rounded_loc[2] + self.vision_size[2],
            ],
            OBS_MY_SIGNAL: self.signal,
            OBS_FREE_STORAGE_PERCENT:
                (self.storage_capacity - len(self.storage)) / self.storage_capacity,
            OBS_HEALTH: self.health,
            OBS_REWARD: self.reward
        }

    def egocentric_r(self, env, obs, a):
        return self.reward

    def egocentric_done(self, env):
        return False

    def egocentric_info(self, env):
        return {
            "SIGNAL": self.signal
        }

    def apply_action(self, a, env):
        """attempts to apply an already
        decided action to environment"""

        a_cont = a[ACT_CONTINUOUS]
        a_signal = a[ACT_SIGNAL]

        # convert possibly tensors into numpy arrays
        if isinstance(a_cont, tf.Variable):
            a_cont = a_cont.numpy()
        if isinstance(a_signal, tf.Variable):
            a_signal = a_signal.numpy()

        # Negative values allow agents to directly 
        # move backward and anti pick or place. Values
        # greator than one would allow running faster
        # than the max speed
        assert np.zeros_like(a_cont) <= a_cont \
            <= np.ones_like(a_cont)

        # make sure signal is valid
        assert 0 <= a_signal < VOCAB_SIZE

        # try to move
        delta_loc = self.max_forward_speed \
            * a_cont[ACT_FORWARD_SPEED_INDEX] \
            * self._dir_vec 
        self.try_move(delta_loc, env)

        # try rotating
        self.orientation += (np.pi/2) * (
            a_cont[ACT_TURN_LEFT_INDEX] \
            - a_cont[ACT_TURN_RIGHT_INDEX])

        # if pick up
        if a_cont[ACT_PICK_INDEX] > 0.5:
            self._pick(env)

        # if place down
        if a_cont[ACT_PLACE_INDEX] > 0.5:
            self._place(env)

        # if consume
        if a_cont[ACT_EAT_INDEX] > 0.5:
            # chewing takes effort whether it has nutrition or not
            self.energy -= EATING_COST
            # consume food if in front of agent
            if OPERATIONS.EAT in self._block_ops_in_front(env):
                self.energy += FOOD_ENERGY
                # remove food block from env
                self.env.static_objects \
                    [self.loc+self._dir_vec] \
                    = OPERATIONS.encode([OPERATIONS.GOTHROUGH])
                self.env.combined_object_ops \
                    [self.loc+self._dir_vec] \
                    = OPERATIONS.encode([OPERATIONS.GOTHROUGH])

        # make signals
        self.set_signal(a_signal)

        # logic
        self.energy -= RESTING_ENERGY_RATE
        if self.energy <= 0:
            env.remove_agent(self)

        return # for clarity

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
                high=VOCAB_SIZE-1,
                shape=self.vision_size,
                dtype=np.int16
            ),
            OBS_MY_SIGNAL: gym.spaces.Discrete(VOCAB_SIZE),
            OBS_FREE_STORAGE_PERCENT: gym.spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float
            ),
            OBS_HEALTH: gym.spaces.Box(
                low=0,
                high=1,
                shape=(1,),
                dtype=np.float
            ),
            OBS_REWARD: gym.spaces.Box(
                low=-1,
                high=1,
                shape=(1,),
                dtype=np.float
            ),
        })

    @property
    def action_space(self):
        return gym.spaces.Dict({
            ACT_CONTINUOUS: gym.spaces.Box(
                low=0,
                high=1,
                shape=(ACT_CONTINUOUS_LEN,),
                dtype=np.float
            ),
            ACT_SIGNAL: gym.spaces.Discrete(VOCAB_SIZE)
        })

    @property
    def health(self):
        """health saturates at unity

        health = 1-exp(-energy/50.0)
        """
        return 1-np.math.exp(-self.energy/50.0)

    def try_move(self, delta_loc, env):
        """
        args
            delta_loc: np.ndarray(np.float) will get converted
                to int's (possibly nondetirministically)
        """
        # speed is porportional to health [0, 1)
        delta_loc *= self.health

        # exerting work costs energy
        displacement = np.linalg.norm(delta_loc, ord=2)
        self.energy -= MOVING_ENERGY_COST * displacement * mass

        # below is reminiscent of a=F/m
        mass = 1 + len(self.storage)
        delta_loc /= mass

        super(Actor, self).try_move(delta_loc, env)
        return # for clarity

    def attack(self, energy_loss):
        """inflict attack on actor. The actor's energy becomes
        energy := max(energy - energy_loss, 0)
        Also, self.pain increases by however much energy
        was actually lost.

        args:
            energy_loss: energy to attempt to remove

        return: returns actual energy lost >= 0"""
        actual_energy_loss = min(self.energy, energy_loss)
        self.energy -= actual_energy_loss
        return actual_energy_loss

    def _pick(self, env):
        """attempt to pick up whatever may be in front
        of the agent
        
        identify first match:
        actor -> actor.attack
        pickable object -> add to `self.storage` if not full
        default -> loss extra energy

        always lose a small amount of energy
        """
        loc_in_front = self.loc+self._dir_vec
        possible_actor = env.actor_at(loc_in_front)
        if possible_actor is not None:
            # attack actor
            self.energy += ATTACKING_GAIN_COEF * possible_actor.attack(10.0)
            self.energy -= ATTACKING_COST
        elif OPERATIONS.PICKUP in OPERATIONS.decode(
            env.combined_object_ops[loc_in_front]) \
            and len(self.storage) < self.storage_capacity:
            # pickup object, place in storage,
            # and replace with empty block in environment
            
            # the block at loc_in_front is not an actor
            # but may still be a moving_object or signaling_moving_object
            # instead of simply a static block
            possible_moving_object = env.moving_object_at(loc_in_front)
            possible_signaling_object = env.signaling_object_at(loc_in_front)
            if possible_signaling_object is not None:
                # remove signaling object, but do not change
                # static object where it stood (necesarily a
                # GOTHROUGHable static object already)
                self.storage.append(possible_signaling_object)
            elif possible_moving_object is not None:
                # remove moving object, but do not change
                # static object where it stood (necesarily a
                # GOTHROUGHable static object already)
                self.storage.append(possible_moving_object)
            else:
                self.storage.append(env.static_objects[loc_in_front])
                env.static_objects[loc_in_front] = OPERATIONS.encode([
                    OPERATIONS.GOTHROUGH])
        else:
            # attempting to pick up costs extra energy if failed
            self.energy -= FAILED_PICKUP_COST
        # picking up always costs energy
        self.energy -= PICKUP_COST

    def _place(self, env):
        """attempt to place whatever the actor may be
        holding
        
        lose small amount of health if placed where not allowed"""
        loc_in_front = self.loc+self._dir_vec
        if OPERATIONS.GOTHROUGH in OPERATIONS.decode(
            env.combined_object_ops[loc_in_front]) \
            and len(self.storage) > 0:
            # place last object in self.storage out
            env.static_objects[loc_in_front] = self.storage.pop()
        else:
            # could not place object or nothing to place
            self.energy -= FAILED_PLACE_COST
        self.energy -= PLACE_COST

    @property
    def _dir_vec(self):
        return [
            np.cos(self.orientation),
            np.sin(self.orientation),
            0.0
        ]

    def _block_ops_in_front(self, env):
        """get operations support by block directly
        in front of actor"""        
        block_loc = self.loc + self._dir_vec
        block = env.combined_object_ops[block_loc]
        return OPERATIONS.decode(block)
        
    @property
    def _calc_energy_gain_reward(self):
        """Non-idempotent reward logic here
        - calculates change in energy by prev_energy and energy
        - updates self.reward with tanh(change in energy / 10.0)
        - replaces self.prev_energy with self.energy

        This method should be called once after all actors
        have performed actions and before observations, rewards, etc.
        are collected for the next timestep.

        Currently, self.egocentric_obs calls this method
        """
        energy_gain = self.energy - self.prev_energy
        self.reward = np.math.tanh(energy_gain / 10.0)
        self.prev_energy = self.energy