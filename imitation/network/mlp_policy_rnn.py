import gym
import numpy as np
import tensorflow as tf

from imitation.common import tf_util as U
from imitation.common.running_mean_std import TfRunningMeanStd
from imitation.common.distributions import make_pdtype

class MlpPolicy():
    recurrent = False
    def __init__(self, name, *args, **kwargs):
        with tf.variable_scope(name):
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, padding_size, gaussian_fixed_var=True):
        # assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=(None,)+ob_space.shape)
        
        # init prev placeholders  FXXK THIS SILLY CODE
        # self.expert_ps = tf.placeholder(shape=[None, None, ob_space.shape[0]], dtype=tf.float32, name='expert_ps')
        # self.expert_pa = tf.placeholder(shape=[None, None, 1], dtype=tf.int32, name='expert_pa')
        # expert_pa_one_hot = tf.squeeze(self.expert_pa, axis=-1)
        # expert_pa_one_hot = tf.one_hot(expert_pa_one_hot, depth=act_space.n, axis=-1)
        
        # expert_prev_state = tf.keras.layers.concatenate([self.expert_ps, expert_pa_one_hot], axis=-1)
        self.prev_ob = U.get_placeholder(name="prev_ob", dtype=tf.float32, shape=(None, padding_size, )+ob_space.shape)
        self.prev_ac = U.get_placeholder(name="prev_ac", dtype=tf.int32, shape=(None, padding_size, 1))
        pa_one_hot = tf.squeeze(self.prev_ac, axis=-1)
        pa_one_hot = tf.one_hot(pa_one_hot, depth=ac_space.n, axis=-1)
        
        prev_state = tf.keras.layers.concatenate([self.prev_ob, pa_one_hot], axis=-1)
        
        
        prev_state_gru = tf.keras.layers.GRU(128, name='polprev_state_gru')
        prev_state_dense = tf.layers.dense(inputs=prev_state, units=128, activation=tf.tanh, name='polprev_state_dense') 
        ob_dense = tf.layers.dense(inputs=ob, units=128, activation=tf.tanh, name='polob_dense') 
        
        prev_gruoutput = prev_state_gru(prev_state_dense)
        
        final_ob = tf.keras.layers.concatenate([ob_dense, prev_gruoutput], axis=-1)

        # Observation normalization
        with tf.variable_scope("obfilter"):
            # self.ob_rms = TfRunningMeanStd(shape=ob_space.shape)
            self.ob_rms = TfRunningMeanStd(shape=256)

        last_out = obz = tf.clip_by_value((final_ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        # Critic trained on discriminator rewards
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred = U.dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        # Critic trained on environmental rewards (ER)
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "vffc_ER%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        self.vpred_ER = U.dense(last_out, 1, "vffinal_ER", weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        # Policy network
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(U.dense(last_out, hid_size, "polfc%i"%(i+1), weight_init=U.normc_initializer(1.0)))
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = tf.nn.tanh(U.dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01)))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.constant_initializer(0))
            logstd = tf.maximum(logstd, tf.log(1e-4))  # for numerical stability
            pdparam = tf.concat(values=[mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = U.dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self._act = U.function([stochastic, ob, self.prev_ob, self.prev_ac], [ac, self.vpred, self.vpred_ER])

    def act(self, stochastic, ob, prev_ob, prev_ac):
        ac, vpred, vpred_ER = self._act(stochastic, ob[None], prev_ob, prev_ac)
        return np.squeeze(ac, axis=0), vpred, vpred_ER
        
    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)
    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)
