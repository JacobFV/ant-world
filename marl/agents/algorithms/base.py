class Base_RL_Algorithm:
    def __init__(self, observation_space, action_space):
        self.observation_space = observation_space
        self.action_space = action_space
        
    def act(self, obs): pass

    def train(self, episode, just_one_frame=None):
        """train agent on `episode`
        
        episode: list of (obs, a, r, done, info)
            tuples. The final tuple may have `None` for
            its action and possibly reward
        just_one_frame: None if you want to optimzie
            the entire trajectory or an integer to
            identify the frame. If a frame is specified
            instead of None, the loss is returned. If
            `just_one_frame=None`, optimization happens
            inside this method and nothing is returned
            
        return: returns nothing if just_one_frame is None,
            otherwise, returns {"pred_loss": pred_loss,
            "pol_loss": pol_loss} for that frame"""
        pass

    def save(self, path): pass
    def restore(self, path): pass

class Base_Actor_Critic_Algorithm(Base_RL_Algorithm):
    def __init__(self, **kwargs):
        super(Base_Actor_Critic_Algorithm, self).__init__(**kwargs)

    def q_fn(self, obs, a): pass

    def value(self, obs):
        return self.q_fn(obs, self.act(obs))