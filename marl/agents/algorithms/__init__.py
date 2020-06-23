class Base_RL_Algorithm:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        
    def act(self, obs): pass

    def train(self, episode):
        """episode: list of (obs, a, r, done, info)
        tuples. The final tuple may have `None` for
        its action and possibly reward"""
        pass

    def save(self, path): pass
    def restore(self, path): pass

class Base_Actor_Critic_Algorithm(Base_RL_Algorithm):
    def __init__(**kwargs):
        super(Base_Actor_Critic_Algorithm, self).__init__(**kwargs)

    def q_fn(self, obs, a): pass

    def value(self, obs):
        return self.q_fn(obs, self.act(obs))