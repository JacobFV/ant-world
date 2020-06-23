from .algorithms import Base_RL_Algorithm

class Agent:
    
    def __init__(self, algorithm):
        """Initialize the agent with a
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
        # the algorithm may be a curried function or class
        if not isinstance(algorithm, Base_RL_Algorithm):
            algorithm = algorithm(
                self.observation_space,
                self.action_space
            )
        self.algorithm = algorithm

    #wrapper for `algorithm` methods
    def act(self, obs):
        return self.algorithm.act(obs)
    def train(self, episode):
        self.algorithm.train(episode)
    def save(self, path):
        self.algorithm.save(path)
    def restore(self, path):
        self.algorithm.restore(path)

    #abstract methods implimented by subclasses
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