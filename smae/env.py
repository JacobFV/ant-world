import numpy as np
import tensorflow as tf
import gym
from PIL import Image

from .actor import Actor, VOCAB_SIZE
from .elements import OPERATIONS, Moving_Object, Signaling_Moving_Object

class MA_Gym_Env(gym.Env):

    def __init__(self, actor_ids=[]):
        """
        args:
            actor_ids: list of user supplied keys to
                identify actors. Can simply be a
                range of ints. If actor_id is of type
                `actor.Actor` then it becomes its own
                key. Otherwise, a random actor is created
                for each actor_id

        NOTE: ids can be initialized later or
            even during the simulation and added to
            the environment with `self.add_id(id)`
        """
        self.actors = { }
        for actor_id in actor_ids:
            self.add_actor(actor_id)
        self.origonal_actors = self.actors.copy()

    def reset(self):
        """reset smae environment state
        
        return: returns vectorized observation for agents"""
        return {
            actor_id: actor.egocentric_obs(self)
            for actor_id, actor in self.actors.items()
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
        self.origonal_actors = self.actors.copy()
        for actor_id, a in a_n.items():
            self.origonal_actors[actor_id].apply_action(a, self)
        self._global_update()
        return {
            actor_id: actor.egocentric_obs(self)
            for actor_id, actor in self.origonal_actors.items()
        }, {
            actor_id: actor.egocentric_r(self)
            for actor_id, actor in self.origonal_actors.items()
        }, {
            actor_id: actor.egocentric_done(self)
            for actor_id, actor in self.origonal_actors.items()
        }, {
            actor_id: actor.egocentric_info(self)
            for actor_id, actor in self.origonal_actors.items()
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
            actor_id: actor.observation_space
            for actor_id, actor in self.actors.items()
        })

    @property
    def action_space(self):
        return gym.spaces.Dict({
            actor_id: actor.action_space
            for actor_id, actor in self.actors.items()
        })

    def add_actor(self, actor):
        """new actors created post-initialization
        or midgame can be added here
        
        args:
            actor: actor to add to environment.
                If type is not `actor.Actor`, a
                random Actor is initialized and
                `actor` arg supplied becomes its
                dictionary key
                
        return: returns `actor` if it was an `actor.Actor`
            object, otherwise returns the newly initialized
            `actor.Actor` mapped to by `actor`"""
        actor_id = actor
        if not isinstance(actor, Actor):
            actor = Actor(env=self)
        self.actors[actor_id] = actor
        return actor

    def remove_actor(self, actor_id=None, actor=None):
        """Remove actor fom environment. Actors that die
        midgame also can remove themselves

        args:
            actor_id: key used for actor. Supply one of
                actor_id or actor but not both.
            actor: actor to remove. Supply one of actor_id
                or actor but not both.
        
        return: returns tuple (actor_id, actor) removed"""
        assert actor_id is None or actor is None
        assert not (actor_id is None and actor is None)

        if actor_id is None:
            actor_ids = list(self.actors.keys())
            actors = list(self.actors.values())
            actor_id = actor_ids[actors.index(actor)]
            
        return actor_id, self.actors.pop(actor_id)

    def random_avaliable_loc(self) -> tuple:
        """Find random location in environment to
        teleport to that satisfies env logical constraints 

        return: returns random location"""
        raise NotImplementedError

class SMAE(MA_Gym_Env):

    metadata = {'render.modes': ["rgb", "human"]}
    
    def __init__(self,
        signal_depth: int,
        world_size=(16,16,4),
        static_objects=None,
        gravity=(0,0,-1),
        **kwargs):
        """
        args:
            world_size: n-tuple of world size (can have dim > 2)
            static_objects: 3D np.ndarray (np.int8) specifying
                allowed ops at each point in space. If `None`
                (default) all points in the environment only
                support the OPERATIONS.GOTHROUGH operation.
            gravity: uniform acceleration vector to apply after
                each step (If z-height = 1, verticle gravity has
                no effect). Can also represent wind force
        """

        super(SMAE, self).__init__(**kwargs)

        self.signal_depth = signal_depth
        self.moving_objects = []
        self.signaling_objects = []
        self.world_size = world_size
        self.gravity = gravity
        self.signal_field = np.zeros(world_size, dtype=np.int16)
        self.static_objects = np.ones(world_size, dtype=np.int8) \
            * OPERATIONS.encode([OPERATIONS.GOTHROUGH]) \
            if static_objects is None else static_objects
        self._global_update()

    def moving_object_at(self, loc):
        """returns the moving object (if present)
        at loc. returns `None` if just static objects"""
        for moving_object in self.moving_objects:
            if moving_object.loc == loc:
                return moving_object
        return None

    def signaling_object_at(self, loc):
        """returns the signaling object (if present)
        at loc. returns `None` if just static objects"""
        obj = self.moving_object_at(loc)
        return obj if isinstance(obj, Signalling_Moving_Object) else None

    def actor_at(self, loc):
        """returns the actor (if present)
        at loc. returns `None` if just static objects"""
        obj = self.moving_object_at(loc)
        return obj if isinstance(obj, Actor) else None

    def default_coloring(self, x, y, z):
        """default coloring scheme for smae env render

        Colors pixels by first match:
        ACTOR: blue, intensity detirmined by signal
        SIGNALING but not ACTOR: green, intensity detirmined by signal
        MOVING but not SIGNALING: brown
        STATIC
         - EATable: yellow
         - PICK_UP and PUSH_OVER: red-orange
         - PUSH_OVER: orange
         - PICK_UP: red
         - GOTHROUGHABLE: transparent (in case multiple z layers are stacked)
        DEFAULT (rigid object): black

        args:
            x,y,z: point in self's space to detirmine color for

        return: returns tuple (r,g,b,a) as np.int8 values 0-255 
        """
        # for brevity in the if cases, static_obj is idenitified here
        static_obj = self.static_objects[x,y,z]
        static_obj_ops = OPERATIONS.decode(static_obj)

        moving_obj = self.moving_object_at((x,y,z))
        # big conditional statement per voxel
        if isinstance(moving_obj, Actor):
            # blue, intensity detirmined by signal
            signal = moving_obj.signal() / VOCAB_SIZE
            return [0, 0, int(255*signal), 255]
        elif self.signaling_object_at((x,y,z)):
            # green, intensity detirmined by signal
            signal = moving_obj.signal() / VOCAB_SIZE
            return [0, int(255*signal), 0, 255]
        elif self.moving_object_at((x,y,z)):
            # brown
            return [128, 32, 16, 255]
        # Now the object is presumed to be static
        elif OPERATIONS.EAT in static_obj_ops:
            # yellow
            return [255, 255, 0, 255]
        elif OPERATIONS.PICKUP in static_obj_ops \
            and OPERATIONS.PUSH_OVER in static_obj_ops:
            # red-orange
            return [255, 64, 0, 255]
        elif OPERATIONS.PUSH_OVER in static_obj_ops:
            # orange
            return [255, 128, 0, 255]
        elif OPERATIONS.PICKUP in static_obj_ops:
            # red
            return [255, 0, 0, 255]
        elif OPERATIONS.GOTHROUGH in static_obj_ops:
            # transparent
            return [0, 0, 0, 0]
        else:
            # black
            return [0, 0, 0, 255]

    def render(self, mode="rgb", z_heights=0, coloring=None, blending=None):
        """renders entire environment as bitmap
        stacking z layers on top of white background
        
        args:
            mode: "rgb" or "human". See env.metadata['render.modes']
            z_heights: int or list of ints of z heights to render
            coloring: color mapping function
                (x,y,z)->(r,g,b,a) as np.int8 values 0-255
                if `None`, self.default_coloring is used
            blending: blending mode (default simple_overlay:
                full overlay if alpha > 0). (back, fore) -> img
                for combining multiple numpy images of format
                (height, width, depth) with depth a 4-discreet
                (r,g,b,a) tuple

        return: returns numpy array (mode="rgb") or 
            Image (mode="human") of render
        """
        if not isinstance(z_heights, list):
            z_heights=[z_heights]
        if coloring is None:
            coloring = self.default_coloring
        if blending is None:
            def simple_overlay(back, fore):
                """override back with any nonzero fore alpha
                return: returns 3D numpy array (H,W,D) with
                depth still (r,g,b,a) but alpha is only zero
                if both back and fore have zero alpha"""
                # this broadcasts alpha onto all four channels
                # (r',g',b',a') := a * (r,g,b,a)
                return back[:,:,:] * (255 - fore[:,:,3]) \
                     + fore[:,:,:] * fore[:,:,3]
            blender = overlay

        # start with white background
        layers = [np.zeros(self.world_size[0:1]+(4,), np.int8)]

        # overlay renders from bottom up
        for z in z_heights:
            # this loop can be parallelized
            pixel_rgba = [coloring(x,y,z) for x, y
                in np.ndindex(self.world_size[0:1])]
            layers.append(
                np.array(
                    pixel_rgba,
                    shape=self.world_size[0:1]+(4,),
                    dtype=np.int8))
        
        # blend layers into np_img
        np_img = layers[0]
        for layer in layers[1:]:
            np_img = blending(np_img, layer)

        return {
            "rgb": np_img,
            "human": Image.fromarray(np_img[:,:,0:2], 'RGB')
        }[mode]

    def add_actor(self, actor):
        """new actors created post-initialization
        or midgame can be added here
        
        args:
            actor: actor to add to environment.
                If type is not `actor.Actor`, a
                random Actor is initialized and
                `actor` arg supplied becomes its
                dictionary lookup key
                
        return: returns `actor` if it was an `actor.Actor`
            object, otherwise returns the newly initialized
            `actor.Actor` mapped to by `actor`"""
        actor = super(SMAE, self).add_actor(actor)
        self.moving_objects.append(actor)
        self.signaling_objects.append(actor)
        self._logic_update()
        return actor

    def remove_actor(self, actor_id=None, actor=None):
        """Remove actor fom environment. Actors that die
        midgame also can remove themselves

        args:
            actor_id: key used for actor. Supply one of
                actor_id or actor but not both.
            actor: actor to remove. Supply one of actor_id
                or actor but not both.
        
        return: returns tuple (actor_id, actor) removed"""
        actor_id, actor = super(SMAE, self).remove_actor(
            actor_id=actor_id, actor=actor)
        self.moving_objects.pop(actor)
        self.signaling_objects.pop(actor)
        self._logic_update()
        return actor_id, actor

    def random_avaliable_loc(self) -> tuple:
        """Find random location in environment that
        supports OPERATIONS.GOTHROUGH

        return: returns random location"""
        # recursively shoot until an allowable space is found
        loc = np.random.randint(0, self.world_size)
        return loc \
            if OPERATIONS.GOTHROUGH in OPERATIONS.decode(
                self.combined_object_ops[loc]) \
            else self.random_avaliable_loc()

    def _global_update(self):
        """All moving objects have moved
        and all signaling objects should
        have made their signals by now"""
        # physics
        # perform global motion here
        self._apply_global_acceleration(self.gravity)
        self._logic_update()

    def _apply_global_acceleration(self, accel_vec):
        for moving_object in self.moving_objects:
            moving_object.try_move(accel_vec, self)

    def _logic_update(self):
        """logic. After execution, no motion
        should occur until next step"""
        self._update_combined_object_ops()
        self._update_signal_field()

    def _update_combined_object_ops(self):
        """update self.combined_objects with
        new moving_object locations"""
        self.combined_object_ops = self.static_objects.copy()
        for moving_object in self.moving_objects:
            self.combined_object_ops[moving_object.loc] = \
                moving_object.ops

    def _update_signal_field(self):
        # make self.signal_field all zero
        self.signal_field.fill(0)
        # add signals currently being broadcast
        for signaling_object in self.signaling_objects:
            self.signal_field[
                signaling_object.rounded_loc[0],
                signaling_object.rounded_loc[1],
                signaling_object.rounded_loc[2]
                ] = signaling_object.signal