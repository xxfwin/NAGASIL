import tensorflow as tf


class Discriminator:
    def __init__(self, env):
        """
        :param env:
        Output of this Discriminator is reward for learning agent. Not the cost.
        Because discriminator predicts  P(expert|s,a) = 1 - P(agent|s,a).
        """
        # print("Discriminator info:")
        # print(env.observation_space.shape, env.action_space.n)
        with tf.variable_scope('discriminator'):
            self.scope = tf.get_variable_scope().name
            self.do_ratio = tf.placeholder(dtype=tf.float32)
            # self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.expert_s = tf.placeholder(dtype=tf.float32, shape=[None, 2 * env.observation_space.shape[0] + env.action_space.n])
            self.expert_a = tf.placeholder(dtype=tf.int32, shape=[None])
            expert_a_one_hot = tf.one_hot(self.expert_a, depth=env.action_space.n)
            # add noise for stabilise training
            # expert_a_one_hot += tf.random_normal(tf.shape(expert_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            expert_s_a = tf.concat([self.expert_s, expert_a_one_hot], axis=1)

            # self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None] + list(env.observation_space.shape))
            self.agent_s = tf.placeholder(dtype=tf.float32, shape=[None, 2 * env.observation_space.shape[0] + env.action_space.n])
            self.agent_a = tf.placeholder(dtype=tf.int32, shape=[None])
            agent_a_one_hot = tf.one_hot(self.agent_a, depth=env.action_space.n)
            # add noise for stabilise training
            # agent_a_one_hot += tf.random_normal(tf.shape(agent_a_one_hot), mean=0.2, stddev=0.1, dtype=tf.float32)/1.2
            agent_s_a = tf.concat([self.agent_s, agent_a_one_hot], axis=1)

            with tf.variable_scope('network') as network_scope:
                prob_1 = self.construct_network(input=expert_s_a, do_ratio=self.do_ratio)
                network_scope.reuse_variables()  # share parameter
                prob_2 = self.construct_network(input=agent_s_a, do_ratio=self.do_ratio)

            with tf.variable_scope('loss'):
                loss_expert = tf.reduce_mean(tf.log(tf.clip_by_value(prob_1, 0.01, 1)))
                loss_agent = tf.reduce_mean(tf.log(tf.clip_by_value(1 - prob_2, 0.01, 1)))
                loss = loss_expert + loss_agent
                loss = -loss
                
                # loss_expert = tf.reduce_mean(tf.clip_by_value(prob_1, 0.01, 1))
                # loss_agent = tf.reduce_mean(tf.clip_by_value(prob_2, 0.01, 1))
                # loss = loss_agent - loss_expert
                
                tfprint = tf.print([loss, loss_expert, loss_agent, tf.reduce_mean(prob_1), tf.reduce_mean(prob_2)], "loss is:")
            

                tf.summary.scalar('discriminator', loss)

            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-3)
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4)
            # according to DCGAN, use adam with small learning rate and modest momentum
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.9, epsilon=1e-7)
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.9)

            optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.9, epsilon=1e-6)
            
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
            # according to WGAN, is suggest using RMSProp as optimizer
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5, epsilon=1e-7)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5)
            # optimizer = tf.train.RMSPropOptimizer(learning_rate=1e-5, decay=0.5)
            # optimizer = tf.train.AdamOptimizer(learning_rate=5e-4, epsilon=1e-4)
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, epsilon=1e-5)
            # optimizer = tf.train.AdamOptimizer(learning_rate=5e-6, epsilon=1e-6)
            # optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, epsilon=1e-5)
            # self.train_op = optimizer.minimize(loss)
            with tf.control_dependencies([tfprint]):
                self.train_op = optimizer.minimize(loss)

            self.rewards = tf.log(tf.clip_by_value(prob_2, 1e-10, 1))  # log(P(expert|s,a)) larger is better for agent
            # self.rewards = tf.clip_by_value(prob_2, 0.01, 1.)

    def construct_network(self, input, do_ratio):
        layer_1 = tf.layers.dense(inputs=input, units=128, activation=tf.nn.leaky_relu, name='layer1')
        layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.nn.leaky_relu, name='layer2')
        layer_3 = tf.layers.dense(inputs=layer_2, units=128, activation=tf.nn.leaky_relu, name='layer3')
        dropout_layer = tf.nn.dropout(layer_3, rate = do_ratio)
        # layer_1 = tf.layers.dense(inputs=input, units=128, name='layer1')
        # layer_2 = tf.layers.dense(inputs=layer_1, units=128, name='layer2')
        # layer_3 = tf.layers.dense(inputs=layer_2, units=128, name='layer3')
        prob = tf.layers.dense(inputs=dropout_layer, units=1, activation=tf.sigmoid, name='prob')
        # prob = tf.layers.dense(inputs=layer_3, units=1, activation=tf.sigmoid, name='prob')
        return prob

    def train(self, expert_s, expert_a, agent_s, agent_a):
        # print("Call train")
        return tf.get_default_session().run(self.train_op, feed_dict={self.expert_s: expert_s,
                                                                      self.expert_a: expert_a,
                                                                      self.agent_s: agent_s,
                                                                      self.agent_a: agent_a,
                                                                      self.do_ratio: 0.0})

    def get_rewards(self, agent_s, agent_a):
        return tf.get_default_session().run(self.rewards, feed_dict={self.agent_s: agent_s,
                                                                     self.agent_a: agent_a,
                                                                     self.do_ratio: 0.})

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

