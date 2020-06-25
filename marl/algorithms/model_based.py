from .base import Base_Actor_Critic_Algorithm
import tensorflow as tf
import sonnet as snt

class Model_Based_Algorithm(Base_Actor_Critic_Algorithm):
    """Simple model-based actor critic agent"""

    def save(self, path): pass
    def restore(self, path): pass

    def __init__(
        self,
        discount_factor=0.9,
        state_encoder=None,
        state_size=64,
        **kwargs):
        """
    
        args:
            discount_factor: reward rollout discount factor
            state_encoder: if `None` (Default), uses 
                `observation_space` to build linear state encoder
            state_size: real vector state length
            observation_space: gym.spaces.Space
            action_space: gym.spaces.Space
        """
        super(Model_Based_Algorithm, self).__init__(**kwargs)

        self.discount_factor = discount_factor

        self.state_encoder = state_encoder if \
            state_encoder is not None else \ 
            snt.Sequential([
                snt.Linear(128),
                tf.nn.relu,
                snt.Linear(64),
                tf.nn.swish,
                snt.Linear(state_size)
            ])
        
        self.policy = snt.Sequential([
            snt.Linear(64),
            tf.nn.swish,
            snt.Linear(64),
            tf.nn.swish,
            snt.Linear(self.action_space.shape[0])
        ])

        self.reward_estimator = snt.Sequential([
            snt.Linear(32),
            tf.nn.swish,
            snt.Linear(16),
            snt.Sum(),
        ])
        
        self.predictor = snt.Sequential([
            snt.Linear(64),
            tf.nn.swish,
            snt.Linear(64),
            tf.nn.swish,
            snt.Linear(state_size)
        ])
    
    def act(self, obs=None, s=None):
        """supply either a direct observation OR state but not both"""
        assert obs is not None or s is not None

        if s is not None:
            assert obs is None
            if s.size == 1:
                s = tf.expand_dims(s, 0) #create a batch of size 1
        else:
            assert obs is not None
            if obs.size == 1:
                obs = tf.expand_dims(obs, 0) #create a batch of size 1
            s = self.state_encoder(obs)
        action = self.policy(s)
        return action[0] #this is the only output in the batch

    def pred(self, s, a):
        s_a_pair = tf.concat([s, a], -1)
        if s_a_pair.size == 1:
            s_a_pair = tf.expand_dims(s_a_pair, 0)
        pred = self.predictor(s_a_pair)
        return pred[0] #this is the only output in the batch

    def estimate_reward(self, s, a):
        s_a_pair = tf.concat([s, a], -1)
        if s_a_pair.size == 1:
            s_a_pair = tf.expand_dims(s_a_pair, 0)
        return self.reward_estimator(s_a_pair)[0]

    def _imag_rollout(self, s, a, T=10):
        """generates rollout for T steps
        
        return: returns list of imagined (s, a) tuples
                INCLUDING the given (s, a) pair so
                len(returned sequence) == T+1"""
        imag_s = [s]
        imag_a = [a]
        for tau in range(T):
            imag_s.append(self.pred(imag_s[-1], imag_a[-1]))
            imag_a.append(self.act(imag_s[-1])) #the last imag_a computation is superfluous
        return [{"s": i_s, "a": i_a}
                for i_s, i_a in zip(imag_s, imag_act)]

    def q_fn(self, s=None, a=None, T=10, rollout=None):
        """computes Q-value of (obs,a) for T steps.
        alternatively, you can supply a precomputed rollout
        (but not both (obs,a) T and rollout)"""
        if rollout is None:
            assert s is not None and a is not None
            rollout = self._imag_rollout(s, a, T)
        else:
            assert s is None and a is None

        discounted_sum = 0.
        for tau, step in enumerate(rollout, start=0):
            discounted_sum += (self.discount_factor ** tau) * \
                self.estimate_reward(step["s"], step["a"])
        return discounted_sum

    def train(self, episode, batch_optimize=True):
        """train agent on `episode`
        
        episode: list of (obs, a, r, done, info)
            tuples. The final tuple may have `None` for
            its action and possibly reward
        batch_optimize: whether to run the optimizer
            for each time step individually or in batch
            (if you want minibatches, right now you will
            have to manually partition the episode.)
            
        return: returns nothing"""

        pred_optimizer = tf.optimizers.Adadelta(learning_rate=0.001)
        pol_optimizer = tf.optimizers.Adadelta(learning_rate=0.001)
        def optimize(pred_loss, pol_loss)
            """optimzie trainable variables with respect to losses

            args:
                pred_loss: (1,) tensor
                pol_loss: (1,) tensor

            return: returns nothing
            """
            pred_min_op = pred_optimizer.minimize(
                loss=pred_loss, var_list=[
                    self.predictor.trainable_variables,
                    self.state_encoder.trainable_variables,
                    self.state_decoder.trainable_variables,
                    self.reward_estimator.trainable_variables])
            pol_min_op = pred_optimizer.minimize(
                loss=pred_loss,
                var_list=self.policy.trainable_variables)

            for _ in range(10):
                pred_min_op.run()
                pol_min_op.run()

            return

        c_recon = 0.2 # reconstructive accuracy importance
        c_pred_roll = 1.0 # predictive rollout accuracy importance
        c_r = 0.5 # reward function accuarcy importance
        c_q_fn = 1.0 # Q function accuracy importance

        pred_loss = []
        pol_loss = []

        # dynamic_s will backpropagate gradients into
        # self.state_encoder unlike frozen_state_seq_enc
        # this actually takes advantage of the batch
        # computation nature of sonnet modules
        dynamic_s = self.encoder(
            [obs for obs, _, _, _, _ in episode])
        # frozen here because I don't want to
        # backpropagate gradients which may encourage
        # the state encoder to lie to optimize
        frozen_state_seq_enc = dynamic_s.numpy()

        t_seq = range(len(episode))
        for t in t_seq:
            obs, a, r, _, _ = episode[t]
            rollout_len = 10
            rollout = self._imag_rollout(
                # s will eventually be transformed into
                # a batch, so it is left in 2D space by
                # slicing instead of indexing
                s=dynamic_s[t:t+1], 
                a=a,
                T=rollout_len)
            pred_loss.append(tf.Variable(0.))
            pol_loss.append(tf.Variable(0.))

            # minimize predictive state trajectory deviation
            @tf.function
            def frechet_dist(true_seq, pred_seq):
                """computes frechet distance between
                sequences with length of shortest sequence"""
                distances = [
                    tf.keras.losses.mse(
                        y_true=true_seq_elem,
                        y_pred=pred_seq_elem
                    )
                    for true_seq_elem, pred_seq_elem
                    in zip(true_seq, pred_seq)]
                beta = 1.e2
                return tf.reduce_sum(tf.nn.softmax(
                    beta*distances), axis=-1)
            pred_loss[-1] += c_pred_roll * frechet_dist(
                true_seq=frozen_state_seq_enc[t:],
                pred_seq=[step["s"] for step in rollout]
            )

            # maximize immediate reward estimation accuracy
            pred_loss[-1] += c_r * tf.keras.losses.mse(
                y_true=r,
                y_pred=self.estimate_reward(
                    frozen_state_seq_enc[t:t+1], a)
            )
            
            # directly maximize Q-function with r-sequence
            # only looking to compare imagined seuence
            # elements against real data so
            # rollout_to_end is clipped such that
            # rollout_to_end[n]["s"] == episode[t+n]["s"]
            # however not necesarily
            # rollout_to_end[-n]["s"] ==* episode[-n]["s"]
            # because the rollout (default len=10)
            # may not predict to the end of the true episode
            rollout_to_end = rollout[t:]
            if t+len(rollout_to_end) > len(episode):
                # in this case, the rollout does reach to the end
                # rollout_to_end[-n]["s"] == episode[-n]["s"]
                # actually that is also valid for rollouts
                # that stretch exactly to the end of the episode
                # but no terminal truncation is necesary then
                rollout_to_end = rollout_to_end \
                    [:len(episode)-(len(rollout_to_end)+t)]
            pred_loss[-1] += c_q_fn * tf.keras.losses.mse(
                y_true=sum([r
                            for _, _, r, _, _
                            in episode[t:]]),
                y_pred=self.q_fn(rollout=rollout_to_end)
            )

            # maximize Q-function over policy space
            pol_loss[-1] = -self.q_fn(rollout=rollout)

            if not batch_optimize:
                # optimize single frame
                optimize(pred_loss[-1], pol_loss[-1])

        if batch_optimize:
            # optimize sum of losses
            optimize(sum(pred_loss), sum(pol_loss))
        
        return