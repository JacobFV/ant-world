import numpy as np
from enum import Enum

class OPERATIONS(enum):
    GOTHROUGH = 0
    PICKUP = 1
    PUSH_OVER = 2

class Moving_Object:
    def __init__(self, loc: list, ops):
        """
        args
            ops: np.int8 bitfield specifying allowed
                operations on object
        """
        self.loc = loc
        self.ops = ops
    
    def try_move(self, delta_loc, env):
        raise NotImplementedError() #TODO


class Signalling_Object:
    def __init__(self, signal_depth=3):
        self.signal = np.zeros((signal_depth,))
    
    @property
    def signal(self):
        return self.signal

    def _set_signal(self, signal):
        self.signal = signal