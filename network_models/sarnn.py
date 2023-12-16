import tensorflow as tf
from tensorflow import keras as K

tf_sess = tf.get_default_session()
K.backend.set_session(tf_sess)

class SARNN:
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
            self.prev_obs = tf.placeholder(shape=[None, None, ob_space.shape[0]], dtype=tf.float32, name='prev_obs')
            self.prev_acts = tf.placeholder(shape=[None, None, 1], dtype=tf.int32, name='prev_acts')
            
            self.tar_ob = tf.placeholder(shape=[None, ob_space.shape[0]], dtype=tf.float32, name='tar_ob')
            self.tar_act = tf.placeholder(shape=[None, 1], dtype=tf.int32, name='tar_act')
            
            prev_acts_one_hot = tf.squeeze(self.prev_acts, axis=-1)
            prev_acts_one_hot = tf.one_hot(prev_acts_one_hot, depth=act_space.n, axis=-1)
            
            tar_act_one_hot = tf.squeeze(self.tar_act, axis=-1)
            tar_act_one_hot = tf.one_hot(tar_act_one_hot, depth=act_space.n, axis=-1)
            # prev_acts_one_hot = tf.squeeze(prev_acts_one_hot, axis=-2)
            
            prev_state = tf.keras.layers.concatenate([self.prev_obs, prev_acts_one_hot], axis=-1)
            
            # obs_gru = tf.keras.layers.GRU(64)
            # act_gru = tf.keras.layers.GRU(16)
            
            # obs_dense = tf.layers.dense(inputs=self.obs, units=128, activation=tf.tanh) 
            
            # state_gru = tf.keras.layers.GRU(64)
            # state_gru = tf.keras.layers.GRU(128)
            
            state_gru = tf.keras.layers.GRU(256, name='prev_rnn')
            
            prev_state_dense = tf.layers.dense(inputs=prev_state, units=256, activation=tf.nn.leaky_relu, name='prev_dense') 
            
            # prev_obs_dense = tf.layers.dense(inputs=self.prev_obs, units=128, activation=tf.tanh)
            
            # obs_gruoutput = obs_gru(prev_obs_dense)
            # act_gruoutput = act_gru(self.prev_acts)
            prev_state_gruoutput = state_gru(prev_state_dense)
            
            # decode rnn result to original dimension
            d_prev_ob_gruoutput = tf.layers.dense(inputs=prev_state_gruoutput, units=ob_space.shape[0], activation=tf.nn.leaky_relu, name='rnn_decode_ob') 
            d_prev_act_gruoutput = tf.layers.dense(inputs=prev_state_gruoutput, units=act_space.n, activation=tf.nn.leaky_relu, name='rnn_decode_act') 
            
            # spilit output to state and softmax(action)
            self.gru_states = d_prev_ob_gruoutput
            # gru_actions = d_prev_state_gruoutput[:, ob_space.shape[0]:]
            # actions should pass softmax
            self.gru_actions = tf.nn.softmax(d_prev_act_gruoutput)
            
            # final_obs = tf.keras.layers.concatenate([self.obs, obs_gruoutput, act_gruoutput], axis=-1)
            # final_obs = tf.keras.layers.concatenate([obs_dense, obs_gruoutput, act_gruoutput], axis=-1)
            
            # ORIGINAL S,A train
            # final_output = tf.keras.layers.concatenate([self.gru_states, self.gru_actions], axis=-1)
            # final_target = tf.keras.layers.concatenate([self.tar_ob, tar_act_one_hot], axis=-1)
            
            # A only train
            final_output = self.gru_actions
            final_target = tar_act_one_hot

            mae = tf.keras.losses.MeanAbsoluteError()
            mse = tf.keras.losses.MeanSquaredError()

            
            # loss_expert = mae(1.0, prob_1)
            # loss_agent = mae(agent_scale, prob_2)
            # loss_expert = mse(1.0, prob_1)
            # loss_agent = mse(agent_scale, prob_2)
            loss = mse(final_output, final_target)
            # loss = mae(final_output, final_target)
            
            tfprint = tf.print([loss], "RNN loss is:")
            
            # optimizer = tf.train.AdamOptimizer()
            # optimizer = tf.train.AdamOptimizer(learning_rate=1e-4, beta1=0.9, epsilon=1e-5)
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
        
    def train(self, prev_obs, prev_acts, tar_ob, tar_act):
        return tf.get_default_session().run(self.train_op, feed_dict={
        self.prev_obs: prev_obs,
        self.prev_acts: prev_acts,
        self.tar_ob: tar_ob,
        self.tar_act: tar_act})

    def get_action_prob(self, prev_obs, prev_acts):
        return tf.get_default_session().run(self.gru_actions, feed_dict={
        self.prev_obs: prev_obs,
        self.prev_acts: prev_acts})
    

