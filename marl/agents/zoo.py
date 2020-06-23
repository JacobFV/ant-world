from .moving_agnet import Moving_Agent

class Zoo:

    def __init__(self, algorithm):
        """Initialize an agent creator with a
        specified intelligence algorithm

        args: 
            algorithm: algorithms.Base_RL_Algorithm
                to govern agent actions and training

                This is the chicken-and-egg problem
                since the algorithms.Base_RL_Algorithm
                needs observation and action spaces to
                initialize also so algorithm may be a
                partially curried function or a class

        return: returns initialized self"""
        self.algorithm = algorithm

    def ant(self, tribe='A'):
        """ant agent initializer

        args:
            tribe: any character A-Z

        return: returns ant
        """
        pass

    #def caterpillar(self):
    #    """caterpillar agent initializer
    #
    #    return: returns caterpillar
    #    """
    #    pass
    #
    #def butterfly(self):
    #    """butterfly agent initializer
    #
    #    return: returns butterfly
    #    """
    #    pass