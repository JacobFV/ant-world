import numpy as np
from enum import Enum

class OPERATIONS(Enum):
    GOTHROUGH = 0 # exclusive of PUSH_OVER or PICKUP
    PICKUP = 1 # exclusive of GOTHROUGH
    PUSH_OVER = 2 # exclusive of GOTHROUGH
    EAT = 3 # exclusive of GOTHROUGH
    # four more operations could be supported with np.int8

    @staticmethod
    def encode(ops: list) -> np.int8:
        """convert a list of OPERATION enums into an int8

        args:
            ops: list of OPERATIONS to encode

        return: returns int8 encoding
        """
        return sum([
            2 ** i
            for i in range(8)
            if OPERATIONS(i) in ops
        ])

    @staticmethod
    def decode(ops_int: np.int8) -> list:
        """decode an int8 to a list of OPERATION enums

        args:
            ops_int: int8 encoding

        return: returns list of OPERATIONS
        """
        return [
            OPERATIONS(i)
            for i
            in range(8)
            if (2**i) & ops_int
        ]

    @staticmethod
    def contains(ops_int, op):
        return bool((2**i) & ops_int)

class Moving_Object:
    def __init__(self,
        loc: list,
        ops=[OPERATIONS.PUSH_OVER, OPERATIONS.PICKUP]):
        """
        args
            loc: initial location tuple
            ops: either list of OPERATIONS or np.int8
                bitfield encoding specifying allowed
                operations on object
        """
        if isinstance(ops, list):
            ops = OPERATIONS.encode(ops)

        self.loc = loc
        self.ops = ops
    
    def try_move(self, delta_loc, env):
        """attempts to move self.loc by delta_loc
        provided no object in the environment cannot be
        pushed over (OPERATIONS.PUSH_OVER) or walked through
        (OPERATIONS.GO_THROUGH)

        args
            delta_loc: np.ndarray(np.float)
                will get converted to int's
                (possibly nondetirministically)
        """
        unit_delta=delta_loc/np.linalg.norm(delta_loc, ord=1)
        target_displ = np.linalg.norm( \
            delta_loc + np.mod(self.loc, 1.0), ord=1)
        # remove decimal location to begin, add back later
        self.loc = np.mod(self.loc, 1.0)
        # step until at target or block is not traversable
        for i in range(int(target_displ)):
            block_loc = self.loc + i*unit_delta
            block = env.combined_object_ops[block_loc]
            block_ops = OPERATIONS.decode(block)
            if OPERATIONS.GOTHROUGH in block_ops:
                # yes, moving is allowed
                pass
            elif OPERATIONS.PUSH_OVER in block_ops:
                # see if block can be pushed over by looking
                # at the space immediantly in travel direction
                # of that block
                next_space = env.combined_object_ops \
                    [block_loc+unit_delta]
                if OPERATIONS.GOTHROUGH in OPERATIONS.decode(next_space):
                    # if that block is actually a 
                    # `moving_object` have it move itself
                    moved_itself = False
                    for mov_obj in env.moving_objects:
                        if mov_obj.loc == block_loc:
                            moved_itself = True
                            mov_obj.try_move(unit_delta,env)
                    # otherwise manually move block over
                    # replace with GO_THROUGHable block
                    if not moved_itself:
                        env.static_objects[block_loc] = \
                            OPERATIONS.encode([OPERATIONS.GOTHROUGH])
                        env.static_objects[next_space] = block
                else:
                    # the space after the object being pushed
                    # over is occupied, so that object cannot
                    # move to allow the actor to move
                    # motion cannot continue
                    break
            else:
                # this space is not GO_THROUGHable 
                # nor can is be PUSH_OVERed
                # the motion cannot continue
                break
            # increment location by one unit 
            # in the direction of delta_loc
            self.loc += unit_delta
        # there may be a decimal remainder not attempted
        # which is just stored in self.loc and added to the
        # next try_move call
        self.loc += target_displ % 1.0
        return # for clarity

    @property
    def rounded_loc(self) -> tuple:
        """get nearest whole number rounded location
        This allows grid-world interactions while not losing
        the ability to perform decimal valued motions"""
        return tuple([round(loc_i) for loc_i in list(self.loc)])

class Signaling_Moving_Object(Moving_Object):
    def __init__(self, signal_depth: int, **kwargs):
        """Create signalling object.

        args
            signal_depth: int. How many float
                values to dedicate for the signal
        """
        super(Signaling_Moving_Object, self).__init__(**kwargs)
        self._signal = np.zeros((signal_depth,))
    
    @property
    def signal(self):
        """get current signal of `self`
        
        return: returns np.ndarry (np.float)
            of current signal
        """
        return self._signal

    def set_signal(self, signal):
        """set current signal of `self`
        
        args:
            signal: np.ndarry (np.float) to signal
        """
        self._signal = signal