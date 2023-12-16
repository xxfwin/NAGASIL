import tensorflow as tf


class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = env.observation_space
        act_space = env.action_space
        self.act_space = act_space
        self.ob_space = env.observation_space
        
        # print(act_space.n)

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None] + list(ob_space.shape), name='obs')
            self.exp_act_probs = tf.placeholder(dtype=tf.float32, shape=[None, act_space.n], name='exp_act_probs')
            self.neg_act_probs = tf.placeholder(dtype=tf.float32, shape=[None, act_space.n], name='neg_act_probs')

            with tf.variable_scope('policy_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=128, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_3, units=act_space.n, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                layer_1 = tf.layers.dense(inputs=self.obs, units=128, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            # choose act involving expRNN and negRNN
            self.act_probs = tf.nn.softmax(self.act_probs + self.exp_act_probs - self.neg_act_probs)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=act_space.n)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, exp_act_probs, neg_act_probs, stochastic=True):
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs, self.exp_act_probs: exp_act_probs, self.neg_act_probs: neg_act_probs})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs, self.exp_act_probs: exp_act_probs, self.neg_act_probs: neg_act_probs})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

