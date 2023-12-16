import tensorflow as tf
import copy


class PPOTrain:
    def __init__(self, Policy, Old_Policy, gamma=0.95, clip_value=0.2, c_1=1., c_2=0.01, c_3=1.):
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
        cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

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
            # loss_exp_act = mae(self.Policy.act_probs, self.exp_act_probs)
            
            # exploss 1
            # loss_exp_act = tf.reduce_mean(tf.exp(tf.abs(tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0)) - tf.log(tf.clip_by_value(self.exp_act_probs, 1e-10, 1.0))) ) )

            # exploss 2
            # exp_ratios = tf.exp(tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0))
                            # - tf.log(tf.clip_by_value(self.exp_act_probs, 1e-10, 1.0)))
            # exp_clipped_ratios = tf.clip_by_value(exp_ratios, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            # loss_exp_act = tf.reduce_mean(exp_clipped_ratios)
            
            # exploss 3
            # loss_exp_act = tf.reduce_mean(tf.log(tf.clip_by_value(tf.abs(self.Policy.act_probs - self.exp_act_probs), 1e-10, 1.0) ) )
            # tf.summary.scalar('Exp dist diff', loss_exp_act)

            # clipped mae loss
            ratios_rnn_act = tf.exp(tf.log(tf.clip_by_value(self.Policy.act_probs, 1e-10, 1.0))
                            - tf.log(tf.clip_by_value(self.Old_Policy.act_probs, 1e-10, 1.0)))

            
            # negloss1
            # loss_neg_act_error = tf.reduce_mean( tf.square(tf.clip_by_value(self.Policy.act_probs - self.neg_act_probs, -1, 0)) )

            # CLIP
            # # loss_neg_act = -tf.reduce_mean(self.Policy.act_probs - self.neg_act_probs)
            # clipped_ratios_neg_act = tf.clip_by_value(ratios_rnn_act, clip_value_min=1 - clip_value, clip_value_max=1 + clip_value)
            # loss_neg_act = tf.minimum(tf.multiply(loss_neg_act_error, ratios_rnn_act), tf.multiply(loss_neg_act_error, clipped_ratios_neg_act))
            # loss_neg_act = tf.reduce_mean(loss_neg_act)



            # New neg act error, negloss2
            # loss_neg_act_error = tf.reduce_mean( tf.square(self.neg_act_probs - self.Policy.act_probs + 1) )

            # negloss3  pi-neg > 0 => 1 /  pi > neg => 1
            # neg_target = tf.where(self.Policy.act_probs > self.neg_act_probs, tf.ones_like(self.neg_act_probs), tf.zeros_like(self.neg_act_probs))
            # loss_neg_act_error = -mse(self.Policy.act_probs, neg_target)

            # negloss3 softmax
            # neg_target = tf.where(self.Policy.act_probs > self.neg_act_probs, tf.ones_like(self.neg_act_probs), tf.zeros_like(self.neg_act_probs))
            # # print(self.Policy.act_probs.get_shape().as_list())
            # # print(neg_target.get_shape().as_list())
            # loss_neg_act_error = -mse(self.Policy.act_probs, tf.nn.softmax(neg_target))

            # negloss3 softmax no label
            # neg_target = tf.clip_by_value(self.Policy.act_probs - self.neg_act_probs, clip_value_min=0, clip_value_max=1)
            # # print(self.Policy.act_probs.get_shape().as_list())
            # # print(neg_target.get_shape().as_list())
            # loss_neg_act_error = -mse(self.Policy.act_probs, tf.nn.softmax(neg_target))

            # negloss3 softmax topk
            # topknum = 10
            # neg_target_topk, _ = tf.math.top_k(self.Policy.act_probs - self.neg_act_probs, k=topknum)
            # neg_target_topk_mat = tf.reshape(neg_target_topk[:, -1], [-1, 1])
            # neg_target = tf.where(self.Policy.act_probs - self.neg_act_probs >= neg_target_topk_mat, tf.ones_like(self.neg_act_probs), tf.zeros_like(self.neg_act_probs))
            # loss_neg_act_error = -mse(self.Policy.act_probs, tf.nn.softmax(neg_target))


            # negloss3 topk filtered actions
            # topknum = 10
            # neg_target_topk, _ = tf.math.top_k(self.Policy.act_probs - self.neg_act_probs, k=topknum)
            # neg_target_topk_mat = tf.reshape(neg_target_topk[:, -1], [-1, 1])
            # # pick top number of actions as label
            # neg_target = tf.where(self.Policy.act_probs - self.neg_act_probs >= neg_target_topk_mat, tf.ones_like(self.neg_act_probs), tf.zeros_like(self.neg_act_probs))

            # # only consider label 1 actions, other actions will not be considered
            # filtered_actions = self.Policy.act_probs * neg_target

            # # softmax
            # # loss_neg_act_error = -mse(filtered_actions, tf.nn.softmax(neg_target))
            # # non softmax
            # loss_neg_act_error = -mse(filtered_actions, neg_target)


            # negloss3 random topk filtered actions
            # topknum = 10
            # neg_target_value = tf.clip_by_value(self.Policy.act_probs - self.neg_act_probs, clip_value_min=1e-10, clip_value_max=1)
            # neg_target_sample = tf.multinomial(tf.log(neg_target_value), num_samples=topknum)
            # neg_target_sample_one_hot = tf.one_hot(neg_target_sample, depth=self.Policy.act_space.n)
            # # print("==========DEBUG SHAPES=========")
            # # print(neg_target_sample.get_shape().as_list())
            # # print(neg_target_sample_one_hot.get_shape().as_list())
            # neg_target = tf.reduce_sum(neg_target_sample_one_hot, axis=-2)
            # # print(neg_target.get_shape().as_list())

            # # only consider label 1 actions, other actions will not be considered
            # filtered_actions = self.Policy.act_probs * neg_target

            # # softmax
            # # loss_neg_act_error = -mse(filtered_actions, tf.nn.softmax(neg_target))
            # # non softmax
            # loss_neg_act_error = -mse(filtered_actions, neg_target)


            # negloss3 random topk filtered actions with negative actions
            # topknum = 20
            # neg_target_value = tf.clip_by_value(self.Policy.act_probs - self.neg_act_probs, clip_value_min=1e-10, clip_value_max=1)
            # neg_target_value_neg = tf.clip_by_value(self.neg_act_probs - self.Policy.act_probs, clip_value_min=1e-10, clip_value_max=1)
            # neg_target_sample = tf.multinomial(tf.log(neg_target_value), num_samples=topknum)
            # neg_target_sample_one_hot = tf.one_hot(neg_target_sample, depth=self.Policy.act_space.n)

            # neg_target_sample_neg = tf.multinomial(tf.log(neg_target_value_neg), num_samples=topknum)
            # neg_target_sample_one_hot_neg = tf.one_hot(neg_target_sample_neg, depth=self.Policy.act_space.n)
            # # print("==========DEBUG SHAPES=========")
            # # print(neg_target_sample.get_shape().as_list())
            # # print(neg_target_sample_one_hot.get_shape().as_list())
            # neg_target = tf.reduce_sum(neg_target_sample_one_hot, axis=-2)
            # neg_target_neg = tf.reduce_sum(neg_target_sample_one_hot_neg, axis=-2)
            # # print(neg_target.get_shape().as_list())

            # # only consider label 1 actions, other actions will not be considered
            # filtered_actions_pos = self.Policy.act_probs * neg_target
            # filtered_actions_neg = self.Policy.act_probs * neg_target_neg
            # filtered_actions = tf.add(filtered_actions_pos, filtered_actions_neg)

            # # softmax
            # # loss_neg_act_error = -mse(filtered_actions, tf.nn.softmax(neg_target))
            # # non softmax
            # loss_neg_act_error = -mse(filtered_actions, neg_target)


            # negloss3 random topk filtered actions negative actions only
            topknum = 10
            # topknum = 1
            # 10% of actions will be considered as negative action
            # topknum = int(self.Policy.act_space.n * 0.1)
            neg_target_value = tf.clip_by_value(self.Policy.act_probs - self.neg_act_probs, clip_value_min=1e-10, clip_value_max=1)
            neg_target_value_neg = tf.clip_by_value(self.neg_act_probs - self.Policy.act_probs, clip_value_min=1e-10, clip_value_max=1)

            neg_target_sample_neg = tf.multinomial(tf.log(neg_target_value_neg), num_samples=topknum)
            neg_target_sample_one_hot_neg = tf.one_hot(neg_target_sample_neg, depth=self.Policy.act_space.n)
            # print("==========DEBUG SHAPES=========")
            # print(neg_target_sample.get_shape().as_list())
            # print(neg_target_sample_one_hot.get_shape().as_list())
            neg_target = tf.reduce_sum(neg_target_sample_one_hot_neg, axis=-2)
            # print(neg_target.get_shape().as_list())

            # only consider label 1 actions, other actions will not be considered
            filtered_actions_neg = self.Policy.act_probs * neg_target
            filtered_actions = filtered_actions_neg

            neg_label = tf.zeros_like(self.neg_act_probs)
            # softmax
            # loss_neg_act_error = -mse(filtered_actions, tf.nn.softmax(neg_target))
            # non softmax
            loss_neg_act_error = -mse(filtered_actions, neg_label)


            # neg loss 4 random topk filtered actions
            # topknum = 10
            # neg_target_value = self.neg_act_probs
            # neg_target_sample = tf.multinomial(tf.log(neg_target_value), num_samples=topknum)
            # neg_target_sample_one_hot = tf.one_hot(neg_target_sample, depth=self.Policy.act_space.n)
            # # print("==========DEBUG SHAPES=========")
            # # print(neg_target_sample.get_shape().as_list())
            # # print(neg_target_sample_one_hot.get_shape().as_list())
            # neg_target = tf.reduce_sum(neg_target_sample_one_hot, axis=-2)
            # # print(neg_target.get_shape().as_list())

            # # only consider label 1 actions, other actions will not be considered
            # filtered_actions = self.Policy.act_probs * neg_target

            # neg_label = tf.zeros_like(self.neg_act_probs)

            # # softmax
            # # loss_neg_act_error = -mse(filtered_actions, tf.nn.softmax(neg_target))
            # # non softmax
            # loss_neg_act_error = -mse(filtered_actions, neg_label)


            # reverse negloss3 softmax
            # neg_target = tf.where(self.Policy.act_probs > self.neg_act_probs, tf.zeros_like(self.neg_act_probs), tf.ones_like(self.neg_act_probs))
            # # print(self.Policy.act_probs.get_shape().as_list())
            # # print(neg_target.get_shape().as_list())
            # loss_neg_act_error = -mse(self.Policy.act_probs, tf.nn.softmax(neg_target))

            loss_neg_act = loss_neg_act_error
            
            # loss_exp_act = tf.reduce_mean(tf.log(tf.clip_by_value(tf.abs(self.Policy.act_probs - self.exp_act_probs), 1e-10, 1.0) ) )
            # # loss_neg_act = tf.reduce_mean(tf.log(tf.clip_by_value(tf.abs(self.Policy.act_probs - self.neg_act_probs), 1e-10, 1.0) ) )
            # loss_neg_act = tf.reduce_mean(tf.log(-tf.clip_by_value(self.Policy.act_probs - self.neg_act_probs, -1.0, -1e-10) ) )
            # print(self.Policy.act_probs.get_shape().as_list())
            # print(neg_target.get_shape().as_list())
            # print(self.neg_act_probs.get_shape().as_list())
            diff_value = tf.reduce_mean(tf.reduce_sum(tf.abs(self.Policy.act_probs - self.neg_act_probs), axis=-1))
            diff_value_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(self.Policy.act_probs - tf.nn.softmax(neg_target)), axis=-1))
            tf.summary.scalar('Neg dist diff', loss_neg_act)
            tf.summary.scalar('Neg dist value', diff_value)
            tf.summary.scalar('Neg dist loss', diff_value_loss)
            tf.summary.scalar('neg_target_sum', tf.reduce_mean(tf.reduce_sum(neg_target, axis=-1)))
            # tf.summary.scalar('neg_target_mean', tf.reduce_mean(neg_target))
            # tf.summary.scalar('neg_target_mean_sm', tf.reduce_mean(tf.nn.softmax(neg_target)))
            
            # construct computation graph for loss
            loss = loss_clip - c_1 * loss_vf + c_2 * entropy + c_3 * loss_neg_act

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

    def train(self, obs, actions, gaes, rewards, v_preds_next, neg_act_probs):
        tf.get_default_session().run(self.train_op, feed_dict={self.Policy.obs: obs,
                                                               self.Old_Policy.obs: obs,
                                                               self.actions: actions,
                                                               self.rewards: rewards,
                                                               self.v_preds_next: v_preds_next,
                                                               self.gaes: gaes,
                                                               self.neg_act_probs: neg_act_probs})

    def get_summary(self, obs, actions, gaes, rewards, v_preds_next, neg_act_probs):
        return tf.get_default_session().run(self.merged, feed_dict={self.Policy.obs: obs,
                                                                    self.Old_Policy.obs: obs,
                                                                    self.actions: actions,
                                                                    self.rewards: rewards,
                                                                    self.v_preds_next: v_preds_next,
                                                                    self.gaes: gaes,
                                                                    self.neg_act_probs: neg_act_probs})

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
