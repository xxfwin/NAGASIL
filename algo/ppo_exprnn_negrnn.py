import tensorflow as tf
import copy


class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1., c_2=0.01, c_3=0.1, c_4=0.1):
        """
        :param Policy:
        :param Old_Policy:
        :param gamma:
        :param clip_value:
        :param c_1: parameter for value difference
        :param c_2: parameter for entropy bonus
        :param c_3: parameter for exp RNN
        """

        self.Policy = Policy
        self.Old_Policy = Old_Policy
        self.gamma = gamma

        pi_trainable = self.Policy.get_trainable_variables()
        old_pi_trainable = self.Old_Policy.get_trainable_variables()
        mae = tf.keras.losses.MeanAbsoluteError()
        mse = tf.keras.losses.MeanSquaredError()

        # assign_operations for policy parameter values to old policy parameters
        with tf.variable_scope('assign_op'):
            self.assign_ops = []
            for v_old, v in zip(old_pi_trainable, pi_trainable):
                self.assign_ops.append(tf.assign(v_old, v))

        # inputs for train_op
        with tf.variable_scope('train_inp'):
            self.actions = tf.placeholder(dtype=tf.int32, shape=[None], name='actions')
            self.rewards = tf.placeholder(dtype=tf.float32, shape=[None], name='rewards')
            self.v_preds_next = tf.placeholder(dtype=tf.float32, shape=[None], name='v_preds_next')
            self.gaes = tf.placeholder(dtype=tf.float32, shape=[None], name='gaes')
            self.exp_act_probs = tf.placeholder(dtype=tf.float32, shape=[None, self.Policy.act_space.n], name='exp_act_probs')
            self.neg_act_probs = tf.placeholder(dtype=tf.float32, shape=[None, self.Policy.act_space.n], name='neg_act_probs')

        act_probs = self.Policy.act_probs
        act_probs_old = self.Old_Policy.act_probs

        # probabilities of actions which agent took with policy
        act_probs = act_probs * tf.one_hot(indices=self.actions, depth=act_probs.shape[1])
        act_probs = tf.reduce_sum(act_probs, axis=1)

        # probabilities of actions which agent took with old policy
        act_probs_old = act_probs_old * tf.one_hot(indices=self.actions, depth=act_probs_old.shape[1])
        act_probs_old = tf.reduce_sum(act_probs_old, axis=1)

        with tf.variable_scope('loss'):
            # construct computation graph for loss_clip
            # ratios = tf.divide(act_probs, act_probs_old)
            ratios = tf.exp(tf.log(tf.clip_by_value(act_probs, 1e-10, 1.0))
                            - tf.log(tf.clip_by_value(act_probs_old, 1e-10, 1.0)))
            clipped_ratios = tf.clip_by_value(ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_clip = tf.minimum(tf.multiply(self.gaes, ratios), tf.multiply(self.gaes, clipped_ratios))
            loss_clip = tf.reduce_mean(loss_clip)
            tf.summary.scalar('loss_clip', loss_clip)

            # construct computation graph for loss of entropy bonus
            entropy = -tf.reduce_sum(self.Policy.act_probs *
                                     tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)), axis=1)
            entropy = tf.reduce_mean(entropy, axis=0)  # mean of entropy of pi(obs)
            tf.summary.scalar('entropy', entropy)

            # construct computation graph for loss of value function
            v_preds = self.Policy.v_preds
            loss_vf = tf.squared_difference(self.rewards + self.gamma * self.v_preds_next, v_preds)
            loss_vf = tf.reduce_mean(loss_vf)
            tf.summary.scalar('value_difference', loss_vf)
            
            # Exp RNN loss 
            # loss_exp_act = mse(self.Policy.act_probs, self.exp_act_probs)
            # # loss_neg_act = mse(self.Policy.act_probs, self.neg_act_probs)
            # # loss_exp_act = mae(self.Policy.act_probs, self.exp_act_probs)
            # # loss_neg_act = mae(self.Policy.act_probs, self.neg_act_probs)
            
            # # loss_neg_act = -tf.reduce_mean(self.Policy.act_probs - self.neg_act_probs)
            # loss_neg_act = tf.reduce_mean( tf.square(tf.clip_by_value(self.Policy.act_probs - self.neg_act_probs, -1, 0)) )

            # Clipped MAE/MSE
            ratios_rnn_act = tf.exp(tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0))
                            - tf.log(tf.clip_by_value(self.Old_Policy.act_probs, 1e-10, 1.0)))

            loss_exp_act_error = mse(self.Policy.act_probs, self.exp_act_probs)
            # loss_exp_act_mae = mae(self.Policy.act_probs, self.exp_act_probs)
            clipped_ratios_exp_act = tf.clip_by_value(ratios_rnn_act, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_exp_act = tf.minimum(tf.multiply(loss_exp_act_error, ratios_rnn_act), tf.multiply(loss_exp_act_error, clipped_ratios_exp_act))
            loss_exp_act = tf.reduce_mean(loss_exp_act)

            
            loss_neg_act_error = tf.reduce_mean( tf.square(tf.clip_by_value(self.Policy.act_probs - self.neg_act_probs, -1, 0)) )
            # loss_neg_act = -tf.reduce_mean(self.Policy.act_probs - self.neg_act_probs)
            clipped_ratios_neg_act = tf.clip_by_value(ratios_rnn_act, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            loss_neg_act = tf.minimum(tf.multiply(loss_neg_act_error, ratios_rnn_act), tf.multiply(loss_neg_act_error, clipped_ratios_neg_act))
            loss_neg_act = tf.reduce_mean(loss_neg_act)

            
            # loss_exp_act = tf.reduce_mean(tf.log(tf.clip_by_value(tf.abs(self.Policy.act_probs - self.exp_act_probs), 1e-10, 1.0) ) )
            # # loss_neg_act = tf.reduce_mean(tf.log(tf.clip_by_value(tf.abs(self.Policy.act_probs - self.neg_act_probs), 1e-10, 1.0) ) )
            # loss_neg_act = tf.reduce_mean(tf.log(-tf.clip_by_value(self.Policy.act_probs - self.neg_act_probs, -1.0, -1e-10) ) )
            
            tf.summary.scalar('Exp dist diff', loss_exp_act)
            tf.summary.scalar('Neg dist diff', loss_neg_act)
            
            # construct computation graph for loss
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy - c_3 * loss_exp_act + c_4 * loss_neg_act

            # minimize -loss == maximize loss
            loss = -loss
            tf.summary.scalar('total', loss)

        self.merged = tf.summary.merge_all()
        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, epsilon=1e-7)
        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-6)
        optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, epsilon=1e-6)
        # optimizer = tf.train.AdamOptimizer(learning_rate=1e-5)
        # optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-5)
        self.gradients = optimizer.compute_gradients(loss, var_list=pi_trainable)
        self.train_op = optimizer.minimize(loss, var_list=pi_trainable)

    def train(self, obs, actions, gaes, rewards, v_preds_next, exp_act_probs, neg_act_probs):
        tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.gaes: gaes,
                                                               self.exp_act_probs: exp_act_probs,
                                                               self.neg_act_probs: neg_act_probs})

    def get_summary(self, obs, actions, gaes, rewards, v_preds_next, exp_act_probs, neg_act_probs):
        return tf.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.actions: actions,
                                                                    self.rewards: rewards,
                                                                    self.v_preds_next: v_preds_next,
                                                                    self.gaes: gaes,
                                                                    self.exp_act_probs: exp_act_probs,
                                                                    self.neg_act_probs: 
                                                                    neg_act_probs})

    def assign_policy_parameters(self):
        # assign policy parameter values to old policy parameters
        return tf.get_default_session().run(self.assign_ops)

    def get_gaes(self, rewards, v_preds, v_preds_next):
        deltas = [r_t + self.gamma * v_next - v for r_t, v_next, v in zip(rewards, v_preds_next, v_preds)]
        # calculate generative advantage estimator(lambda = 1), see ppo paper eq(11)
        gaes = copy.deepcopy(deltas)
        for t in reversed(range(len(gaes) - 1)):  # is T-1, where T is time step which run policy
            gaes[t] = gaes[t] + self.gamma * gaes[t + 1]
        return gaes

    def get_grad(self, obs, actions, gaes, rewards, v_preds_next):
        return tf.get_default_session().run(self.gradients, feed_dict={self.Policy.obs: obs,
                                                                       self.Old_Policy.obs: obs,
                                                                       self.actions: actions,
                                                                       self.rewards: rewards,
                                                                       self.v_preds_next: v_preds_next,
                                                                       self.gaes: gaes})
