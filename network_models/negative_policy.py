import tensorflow as tf
from tensorflow import keras as K

tf_sess = tf.get_default_session()
K.backend.set_session(tf_sess)

class Negative_policy:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = env.observation_space
        act_space = env.action_space

        with tf.variable_scope(name):
            # self.obs = tf.placeholder(dtype=tf.float32, shape=[None, ob_space.shape[0]], name='obs')
            # debugoutput1 = tf.Print(self.obs, [tf.shape(self.obs)], "obs shape: ")
            self.obs = tf.placeholder(shape=[None, 2 * ob_space.shape[0] + act_space.n], dtype=tf.float32, name='prev_obs')
            self.acts = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='prev_acts')
            
            # self.tar_ob = tf.placeholder(shape=[None, ob_space.shape[0]], dtype=tf.float32, name='tar_ob')
            # self.tar_act = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='tar_act')
            acts_one_hot = tf.squeeze(self.acts, axis=-1)
            acts_one_hot = tf.one_hot(acts_one_hot, depth=act_space.n, axis=-1)
            # acts_one_hot = tf.squeeze(acts_one_hot, axis=-2)
            acts_one_hot = tf.reshape(acts_one_hot, [-1, act_space.n])
            print(acts_one_hot.get_shape().as_list())

            
            # spilit output to state and softmax(action)
            # gru_actions = d_prev_state_gruoutput[:, ob_space.shape[0]:]
            # actions should pass softmax
            neg_obs_dense1 = tf.layers.dense(inputs=self.obs, units=512, activation=tf.nn.leaky_relu, name='neg_obs_dense1') 
            neg_obs_dense2 = tf.layers.dense(inputs=neg_obs_dense1, units=512, activation=tf.nn.leaky_relu, name='neg_obs_dense2') 
            neg_obs_dropout = tf.nn.dropout(neg_obs_dense2, keep_prob=0.5)
            # neg_obs_dropout = tf.nn.dropout(neg_obs_dense2, keep_prob=0.8)
            # neg_obs_dropout = tf.nn.dropout(neg_obs_dense2, keep_prob=1.0)
            neg_obs_out = tf.layers.dense(inputs=neg_obs_dropout, units=act_space.n, activation=tf.nn.softmax, name='neg_obs_out') 
            self.neg_actions = neg_obs_out

            # self.gru_actions = tf.nn.softmax(d_prev_act_gruoutput)
            
            # final_obs = tf.keras.layers.concatenate([self.obs, obs_gruoutput, act_gruoutput], axis=-1)
            # final_obs = tf.keras.layers.concatenate([obs_dense, obs_gruoutput, act_gruoutput], axis=-1)
            
            # ORIGINAL S,A train
            # final_output = tf.keras.layers.concatenate([self.gru_states, self.gru_actions], axis=-1)
            # final_target = tf.keras.layers.concatenate([self.tar_ob, tar_act_one_hot], axis=-1)
            
            # A only train
            final_output = neg_obs_out
            final_target = acts_one_hot

            mae = tf.keras.losses.MeanAbsoluteError()
            mse = tf.keras.losses.MeanSquaredError()
            # cce = tf.keras.losses.CategoricalCrossentropy(from_logits=False)

            # DEBUG
            print(final_output.get_shape().as_list())
            print(final_target.get_shape().as_list())
            
            # loss_expert = mae(1.0, prob_1)
            # loss_agent = mae(agent_scale, prob_2)
            # loss_expert = mse(1.0, prob_1)
            # loss_agent = mse(agent_scale, prob_2)
            # loss = mse(final_output, final_target)
            # loss = mae(final_output, final_target)
            loss = tf.reduce_mean(tf.keras.losses.categorical_crossentropy(final_target, final_output))

            diff_value = tf.reduce_mean(tf.reduce_sum(tf.abs(final_output - final_target), axis = -1) )
            # print(diff_value.get_shape().as_list())
            
            tfprint = tf.print([loss, diff_value], "Negative policy loss, sum is:")
            
            # optimizer = tf.train.AdamOptimizer()
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-3, beta1=0.9, epsilon=1e-4)
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, epsilon=1e-5)
            # optimizer = tf.train.AdamOptimizer(learning_rate=5e-5, beta1=0.9, epsilon=1e-5)
            optimizer = tf.train.AdamOptimizer(learning_rate=1e-5, beta1=0.9, epsilon=1e-6)

            with tf.control_dependencies([tfprint]):
                self.train_op = optimizer.minimize(loss)
            # self.train_op = optimizer.minimize(loss)


    # def get_action_prob(self, obs):
        # return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs[0], self.prev_obs: obs[1], self.prev_acts: obs[2]})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        
    def train(self, obs, acts):
        return tf.get_default_session().run(self.train_op, feed_dict={
        self.obs: obs,
        self.acts: acts})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.neg_actions, feed_dict={
        self.obs: obs})
    

