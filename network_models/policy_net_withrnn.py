import tensorflow as tf
from tensorflow import keras as K

tf_sess = tf.get_default_session()
K.backend.set_session(tf_sess)

class Policy_net:
    def __init__(self, name: str, env):
        """
        :param name: string
        :param env: gym env
        """

        ob_space = env.observation_space
        act_space = env.action_space

        with tf.variable_scope(name):
            self.obs = tf.placeholder(dtype=tf.float32, shape=[None, ob_space.shape[0]], name='obs')
            # debugoutput1 = tf.Print(self.obs, [tf.shape(self.obs)], "obs shape: ")
            self.prev_obs = tf.placeholder(shape=[None, None, ob_space.shape[0]], dtype=tf.float32, name='prev_obs')
            self.prev_acts = tf.placeholder(shape=[None, None, 1], dtype=tf.int32, name='prev_acts')
            
            prev_acts_one_hot = tf.squeeze(self.prev_acts, axis=-1)
            prev_acts_one_hot = tf.one_hot(prev_acts_one_hot, depth=act_space.n, axis=-1)
            # prev_acts_one_hot = tf.squeeze(prev_acts_one_hot, axis=-2)
            
            prev_state = tf.keras.layers.concatenate([self.prev_obs, prev_acts_one_hot], axis=-1)
            
            # obs_gru = tf.keras.layers.GRU(64)
            # act_gru = tf.keras.layers.GRU(16)
            
            obs_dense = tf.layers.dense(inputs=self.obs, units=128, activation=tf.tanh) 
            
            # state_gru = tf.keras.layers.GRU(64)
            state_gru = tf.keras.layers.GRU(128)
            
            prev_state_dense = tf.layers.dense(inputs=prev_state, units=128, activation=tf.tanh) 
            
            # prev_obs_dense = tf.layers.dense(inputs=self.prev_obs, units=128, activation=tf.tanh)
            
            # obs_gruoutput = obs_gru(prev_obs_dense)
            # act_gruoutput = act_gru(self.prev_acts)
            prev_state_gruoutput = state_gru(prev_state_dense)
            
            # final_obs = tf.keras.layers.concatenate([self.obs, obs_gruoutput, act_gruoutput], axis=-1)
            # final_obs = tf.keras.layers.concatenate([obs_dense, obs_gruoutput, act_gruoutput], axis=-1)
            final_obs = tf.keras.layers.concatenate([obs_dense, prev_state_gruoutput], axis=-1)

            with tf.variable_scope('policy_net'):
                
                # layer_1 = tf.layers.dense(inputs=final_obs, units=128, activation=tf.nn.leaky_relu)
                # # layer_1 = tf.layers.dense(inputs=debugoutput1, units=20, activation=tf.tanh)
                # # layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                # layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.nn.leaky_relu)
                # layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.nn.leaky_relu)
                
                # layer_1 = tf.layers.dense(inputs=debugoutput1, units=20, activation=tf.tanh)
                # layer_1 = tf.layers.dense(inputs=self.obs, units=20, activation=tf.tanh)
                
                
                layer_1 = tf.layers.dense(inputs=final_obs, units=128, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.tanh)
                layer_3 = tf.layers.dense(inputs=layer_2, units=act_space.n, activation=tf.tanh)
                # layer_3 = tf.layers.dense(inputs=final_obs, units=act_space.n, activation=tf.tanh)
                self.act_probs = tf.layers.dense(inputs=layer_3, units=act_space.n, activation=tf.nn.softmax)

            with tf.variable_scope('value_net'):
                # layer_1 = tf.layers.dense(inputs=final_obs, units=128, activation=tf.nn.leaky_relu)
                # # layer_1 = tf.layers.dense(inputs=self.obs, units=128, activation=tf.tanh)
                # layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.nn.leaky_relu)
                # layer_1 = tf.layers.dense(inputs=self.obs, units=128, activation=tf.tanh)
                
                
                layer_1 = tf.layers.dense(inputs=final_obs, units=128, activation=tf.tanh)
                layer_2 = tf.layers.dense(inputs=layer_1, units=128, activation=tf.tanh)
                self.v_preds = tf.layers.dense(inputs=layer_2, units=1, activation=None)

            self.act_stochastic = tf.multinomial(tf.log(self.act_probs), num_samples=act_space.n)
            self.act_stochastic = tf.reshape(self.act_stochastic, shape=[-1])

            self.act_deterministic = tf.argmax(self.act_probs, axis=1)

            self.scope = tf.get_variable_scope().name

    def act(self, obs, stochastic=True):
        # updated, obs contains three parts: [obs, prev_obs, prev_acts]
        if stochastic:
            return tf.get_default_session().run([self.act_stochastic, self.v_preds], feed_dict={self.obs: obs[0], self.prev_obs: obs[1], self.prev_acts: obs[2]})
        else:
            return tf.get_default_session().run([self.act_deterministic, self.v_preds], feed_dict={self.obs: obs[0], self.prev_obs: obs[1], self.prev_acts: obs[2]})

    def get_action_prob(self, obs):
        return tf.get_default_session().run(self.act_probs, feed_dict={self.obs: obs[0], self.prev_obs: obs[1], self.prev_acts: obs[2]})

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
        
    

