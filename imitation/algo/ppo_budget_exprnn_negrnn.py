import time
from collections import deque
import numpy as np
import tensorflow as tf

from imitation.priority_buffer import TrajReplay
from imitation.common.adam import Adam
from imitation.common import logger, dataset, tf_util as U
from imitation.common.math_util import explained_variance
from imitation.common.misc_util import zipsame, fmt_row, OrderedDefaultDict

import math

np.set_printoptions(precision=3)

EXPERT_SIZE = 20
PADDING_SIZE = 20

NEGATIVE_RATIO = 0.1

def rollout_generator(pi, env, discriminator, timesteps_per_batch, running_paths, stochastic, resultf, total_budget, user_cost, traj_list, traj_reward_list):
    t = 0
    running_path_idx = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    
    new = True  # marks if we're on first timestep of an episode
    new_time_limit = True  # marks if the previous episode ended due to enforced time-limits
    rew_ER = 0.0    # environmental rewards
    ob = env.reset()
    
    # avoid pointer
    running_budget = 0 + total_budget
    N = len(user_cost)
    action_space = range(N)
    
    # resultf = open(resultdir, "a")
    
    # print("ac is :", ac)
    # print("ob is :", ob)

    cur_ep_len = 0  # length of current episode
    cur_ep_ret_ER = 0   # environmental returns in current episode
    ep_lens = []  # lengths of episodes completed in this rollout
    ep_rets_oracle = []  # episodic oracle returns (only used for plotting!)

    # Initialize history arrays
    obs = np.array([ob for _ in range(timesteps_per_batch)])
    rews_ER = np.zeros(timesteps_per_batch, 'float32')
    vpreds = np.zeros(timesteps_per_batch, 'float32')
    vpreds_ER = np.zeros(timesteps_per_batch, 'float32')
    news = np.zeros(timesteps_per_batch, 'int32')
    news_time_limit = np.zeros(timesteps_per_batch, 'int32')
    acs = np.array([ac for _ in range(timesteps_per_batch)])
        
    prev_obs = np.array([np.zeros((PADDING_SIZE, ob.shape[0])) for _ in range(timesteps_per_batch)])
    prev_acs = np.array([np.zeros((PADDING_SIZE, 1)) for _ in range(timesteps_per_batch)])
    prevacs = acs.copy()
    
    print(obs.shape)
    print(acs.shape)

    while True:
    
        infected_user = ob[N : 2*N]
        recovered_user = ob[3*N : 4*N]
        infected_prop_num = ob[0 : N]
        recovered_prop_num = ob[2*N : 3*N]
        num_followers = ob[4*N : 5*N]
        
        user_under_budget = np.zeros(N)
        for i in range(N):
            if running_budget >= user_cost[i]:
                user_under_budget[i] = 1
        
        prevac = ac
        
        # Build up prev states and actions
        if len(running_paths[running_path_idx]['obs']) == 0:
            prev_obs_feed = np.zeros((1, PADDING_SIZE, ob.shape[0]))
            prev_acts_feed = np.zeros((1, PADDING_SIZE, 1))
        else:
            # print(np.array(running_paths[running_path_idx]['obs']).flatten())
            prev_obs_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(running_paths[running_path_idx]['obs']).flatten()], maxlen=ob.shape[0]*PADDING_SIZE, padding='post')
            # print(prev_obs_feed.shape)
            
            prev_acts_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(running_paths[running_path_idx]['acs']).flatten()], maxlen=PADDING_SIZE)

            prev_obs_feed = prev_obs_feed.reshape(1, PADDING_SIZE, ob.shape[0])
            # print(prev_obs_feed.shape)
            
            prev_acts_feed = prev_acts_feed.reshape(1, PADDING_SIZE, 1)
        
        for i in range(N):
            ac, vpred, vpred_ER = pi.act(stochastic, ob)
            if user_under_budget[int(ac)] == 1:
                break
        # model cannnot select under budget user
        if user_under_budget[int(ac)] != 1:
            ac = random.choices(action_space, weights = user_under_budget, k = 1)[0]
        
        
        toEnd = True
        left_budget = running_budget - user_cost[int(ac)]
        for i in range(N):
            # if have users still selectable
            if left_budget >= user_cost[i]:
                toEnd = False
                break
        


        # for the first (s,a) in an episode, will be empty list
        running_paths[running_path_idx]['prev_obs'].append(prev_obs_feed)
        running_paths[running_path_idx]['prev_acs'].append(prev_acts_feed)
        
        running_paths[running_path_idx]['obs'].append(ob)
        running_paths[running_path_idx]['acs'].append(ac)

        # Slight weirdness here because we need value function at time T
        # before returning segment [0, T-1] so we get the correct
        # terminal value
        if t > 0 and t % timesteps_per_batch == 0:

            # Obtain discriminator rewards
            rews = np.squeeze(discriminator.get_reward(obs, acs), axis=1)

            yield {"obs" : obs, "rews" : rews, "rews_ER" : rews_ER, "vpreds" : vpreds, "vpreds_ER" : vpreds_ER, "news" : news,
                    "news_time_limit" : np.append(news_time_limit, new_time_limit), "acs" : acs, "prevacs" : prevacs, "nextvpred": vpred * (1 - new),
                    "nextvpred_ER": vpred_ER * (1 - new), "ep_lens" : ep_lens, "ep_rets_oracle": ep_rets_oracle, "prev_obs": prev_obs, "prev_acs" : prev_acs}
            # Be careful!!! if you change the downstream algorithm to aggregate
            # several of these batches, then be sure to do a deepcopy
            ep_rets_oracle = []
            ep_lens = []

        i = t % timesteps_per_batch
        obs[i] = ob
        acs[i] = ac
        vpreds[i] = vpred
        vpreds_ER[i] = vpred_ER
        news[i] = new
        news_time_limit[i] = new_time_limit

        prev_obs[i] = prev_obs_feed
        prev_acs[i] = prev_acts_feed

        prevacs[i] = prevac
        
        # print("ac is :", ac)
        # print(type(ac))

        
        running_budget = running_budget - user_cost[int(ac)]
        
        ob, rew_ER, new, info = env._step(int(ac), toEnd = toEnd)
        
        
        # new_time_limit = 1 if 'timestep_limit_reached' in info.keys() else 0
        new_time_limit = 0 
        rews_ER[i] = rew_ER
        cur_ep_ret_ER += rew_ER
        cur_ep_len += 1

        if new:
            running_paths[running_path_idx]['return'] = cur_ep_ret_ER
            resultf.write(str(cur_ep_ret_ER) + "\n")
            
            # add current finished episode ob, act and env_return to traj
            traj_list.append([running_paths[running_path_idx]['obs'], running_paths[running_path_idx]['acs']])
            traj_reward_list.append(cur_ep_ret_ER)

            running_path_idx += 1
            ep_rets_oracle.append(info['episode']['r'])  # Retrieve the oracle score inserted by monitor.py
            ep_lens.append(cur_ep_len)
            cur_ep_ret_ER = 0
            cur_ep_len = 0
            ob = env.reset()
            
            
            # reset running budget
            running_budget = 0 + total_budget
        t += 1

def add_vtarg_and_adv(rollout, gamma, lam):
    """
    Calculate, independently for discriminator and environmental rewards, the advantage
    using GAE, and critic-target using TD (lambda)
    """
    new = np.append(rollout["news"], 0)
    new_time_limit = rollout["news_time_limit"]
    vpred = np.append(rollout["vpreds"], rollout["nextvpred"])
    vpred_ER = np.append(rollout["vpreds_ER"], rollout["nextvpred_ER"])
    rew = rollout["rews"]
    rew_ER = rollout["rews_ER"]
    T = len(rew)
    rollout["adv"] = gaelam = np.empty(T, 'float32')
    rollout["adv_ER"] = gaelam_ER = np.empty(T, 'float32')
    lastgaelam = 0; lastgaelam_ER = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        delta_ER = rew_ER[t] + gamma * vpred_ER[t+1] * nonterminal - vpred_ER[t]
        gaelam[t] = lastgaelam = (delta + gamma * lam * nonterminal * lastgaelam) * (1-new_time_limit[t+1])
        gaelam_ER[t] = lastgaelam_ER = (delta_ER + gamma * lam * nonterminal * lastgaelam_ER) * (1-new_time_limit[t+1])
    rollout["tdlamret"] = rollout["adv"] + rollout["vpreds"]
    rollout["tdlamret_ER"] = rollout["adv_ER"] + rollout["vpreds_ER"]

def learn(env, policy_fn, RNN_fn, discriminator, *,
        d_step, timesteps_per_batch,
        clip_param, entcoef, vf_coef,
        optim_epochs, optim_stepsize, optim_batchsize,
        gamma, lam, d_stepsize,
        max_timesteps, schedule, # annealing for stepsize parameters (epsilon and adam)
        data_dump_path, mu, pq_replay_size, resultdir, total_budget, user_cost, iteration_num):

    log_record = U.logRecord(data_dump_path)
    
    resultf = open(resultdir, "w")
    

    # Setup losses and stuff
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_fn("pi", ob_space, ac_space)
    oldpi = policy_fn("oldpi", ob_space, ac_space)
    Exp_RNN = RNN_fn('expert_RNN', env)
    Neg_RNN = RNN_fn('negative_RNN', env)

    # for ExpRNN, NegRNN train, save all episodes' ob, act and reward
    trajs = []
    traj_rewards = []

    atarg = tf.placeholder(dtype=tf.float32, shape=[None]) # Target advantage function (if applicable)
    ret = tf.placeholder(dtype=tf.float32, shape=[None]) # TD (lambda) return with discriminator rewards
    ret_ER = tf.placeholder(dtype=tf.float32, shape=[None]) # TD (lambda) return with environmental rewards
    oldvpred = tf.placeholder(dtype=tf.float32, shape=[None])
    oldvpred_ER = tf.placeholder(dtype=tf.float32, shape=[None])
    exp_act_probs = tf.placeholder(dtype=tf.float32, shape=[None, ac_space.n], name='exp_act_probs')
    neg_act_probs = tf.placeholder(dtype=tf.float32, shape=[None, ac_space.n], name='neg_act_probs')

    lrmult = tf.placeholder(name='lrmult', dtype=tf.float32, shape=[]) # learning rate multiplier, updated with schedule
    clip_param = clip_param * lrmult # Annealed cliping parameter epislon

    ob = U.get_placeholder_cached(name="ob")
    ac = pi.pdtype.sample_placeholder([None])
    # policy_acs = pi.pd.logits
    policy_acs = pi.pdparam

    ent = pi.pd.entropy()
    meanent = U.mean(ent)
    pol_entpen = (-entcoef) * meanent

    logratio = tf.clip_by_value(pi.pd.logp(ac) - oldpi.pd.logp(ac), -20., 20.)  # clip for numerical stability
    ratio = tf.exp(logratio) # pnew / pold
    surr1 = ratio * atarg # surrogate from conservative policy iteration

    surr2 = tf.clip_by_value(ratio, 1.0 - clip_param, 1.0 + clip_param) * atarg
    pol_surr = - U.mean(tf.minimum(surr1, surr2)) # PPO's pessimistic surrogate (L^CLIP)

    # exp RNN loss 
    mae = tf.keras.losses.MeanAbsoluteError()
    mse = tf.keras.losses.MeanSquaredError()
    # print(policy_acs)
    # print(policy_acs.shape)
    # print(exp_act_probs.shape)
    # pol_exp_dist = mae(policy_acs, exp_act_probs)
    pol_exp_dist = mse(policy_acs, exp_act_probs)
    # pol_neg_dist = -mae(policy_acs, neg_act_probs)
    # pol_neg_dist = tf.reduce_mean(policy_acs - neg_act_probs)
    pol_neg_dist = -tf.reduce_mean( tf.square(tf.clip_by_value(policy_acs - neg_act_probs, -1, 0)) )
    


    # Clipped Critic losses
    vpredclipped = oldvpred + tf.clip_by_value(pi.vpred - oldvpred, -tf.abs(oldvpred)*clip_param, tf.abs(oldvpred)*clip_param)
    vf_losses1 = tf.square(pi.vpred - ret)
    vf_losses2 = tf.square(vpredclipped - ret)
    vf_loss = vf_coef * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

    vpredclipped = oldvpred_ER + tf.clip_by_value(pi.vpred_ER - oldvpred_ER, -tf.abs(oldvpred_ER)*clip_param, tf.abs(oldvpred_ER)*clip_param)
    vf_losses1 = tf.square(pi.vpred_ER - ret_ER)
    vf_losses2 = tf.square(vpredclipped - ret_ER)
    vf_ER_loss = vf_coef * tf.reduce_mean(tf.maximum(vf_losses1, vf_losses2))

    total_pi_loss = pol_surr + pol_entpen + pol_exp_dist + pol_neg_dist
    total_vf_loss = vf_ER_loss + vf_loss
    pi_losses = [pol_surr, pol_entpen, meanent, pol_exp_dist, pol_neg_dist]
    vf_losses = [vf_loss, vf_ER_loss]
    pi_loss_names = ["pol_surr", "pol_entpen", "ent", "ExpRNN_dist", "NegRNN_dist"]
    vf_loss_names = ["vf_loss", "vf_ER_loss"]
    loss_names = pi_loss_names + vf_loss_names

    # Separate actor-critic paramters into 2 lists
    var_list = pi.get_trainable_variables()
    pi_var_list = [v for v in var_list if v.name.split("/")[1].startswith("pol")]
    pi_var_list += [v for v in var_list if v.name.split("/")[1].startswith("logstd")]
    vf_var_list = [v for v in var_list if v.name.split("/")[1].startswith("vff")]
    assert not (set(var_list) - set(pi_var_list) - set(vf_var_list)), "Missed variables."

    pi_lossandgrad = U.function([ob, ac, atarg, lrmult, exp_act_probs, neg_act_probs], pi_losses + [U.flatgrad(total_pi_loss, pi_var_list)])
    vf_lossandgrad = U.function([ob, ret, ret_ER, oldvpred, oldvpred_ER, lrmult], vf_losses + [U.flatgrad(total_vf_loss, vf_var_list)])

    # Optimizers for actor, critic and discriminator
    pi_adam = Adam(pi_var_list, epsilon=1e-6)
    vf_adam = Adam(vf_var_list, epsilon=1e-6)
    d_adam = Adam(discriminator.get_trainable_variables(), epsilon=1e-6)

    assign_old_eq_new = U.function([], [], updates=[tf.assign(oldv, newv)
        for (oldv, newv) in zipsame(oldpi.get_variables(), pi.get_variables())])
    compute_pi_losses = U.function([ob, ac, atarg, lrmult, exp_act_probs, neg_act_probs], pi_losses)
    compute_vf_losses = U.function([ob, ret, ret_ER, oldvpred, oldvpred_ER, lrmult], vf_losses)

    # Initiate priority-queue replay
    pq_replay = TrajReplay(capacity=pq_replay_size)

    U.initialize()

    # Prepare for rollouts
    # ----------------------------------------
    running_paths = OrderedDefaultDict()  # dictionary to save competed paths (paths == episodes == trajectories)
    rg = rollout_generator(pi, env, discriminator, timesteps_per_batch, running_paths, stochastic=True, resultf=resultf, total_budget=total_budget, user_cost=user_cost, traj_list=trajs, traj_reward_list=traj_rewards)

    episodes_so_far = 0
    timesteps_so_far = 0
    iters_so_far = 0
    tstart = time.time()
    lenbuffer = deque(maxlen=100) # rolling buffer for episode lengths
    rets_oracle_buffer = deque(maxlen=100)   # rolling buffer for episodic environmental returns

    # == Main loop ==
    while timesteps_so_far < max_timesteps:

        if schedule == 'constant':
            cur_lrmult = 1.0
        elif schedule == 'linear':
            cur_lrmult = max(1.0 - float(timesteps_so_far) / max_timesteps, 0)
        else:
            raise NotImplementedError

        logger.log("[%s] ********** Iteration %i ************" % (time.strftime('%x %X %z'), iters_so_far))

        # ------------------ Update G (policy) ------------------
        logger.log("Optimizing Policy...")

        # Rollout using current policy
        rollout = rg.__next__()
        add_vtarg_and_adv(rollout, gamma, lam)

        # Add completed episodes to priority replay, and sync!
        for path in list(running_paths.values())[:-1]:
            pq_replay.add_path(path)
        pq_replay.sync()

        # use running_path to save all episodes, pick memory episodes from running_path
        # Reclaim memory
        # for k in list(running_paths.keys())[:-1]:
            # del running_paths[k]

        # =================================
        # train ExpRNN
        # sort traj according to env reward
        traj_order = np.argsort(traj_rewards)[::-1]
        expertRNN_prev_obs = None
        expertRNN_prev_acts = None
        expertRNN_tar_ob = None
        expertRNN_tar_act = []
        for i in traj_order[:EXPERT_SIZE]:
            # ===================================================
            # Use every s,a pair as input to train expert rnn
            for ii in range(len(trajs[i][0]) - 1):
                prev_obs_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(trajs[i][0][:ii+1]).flatten()], maxlen=ob_space.shape[0]*PADDING_SIZE, padding='post')
                # print(prev_obs_feed.shape)
                
                prev_acts_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(trajs[i][1][:ii+1]).flatten()], maxlen=PADDING_SIZE)

                prev_obs_feed = prev_obs_feed.reshape(1, PADDING_SIZE, ob_space.shape[0])
                # print(prev_obs_feed.shape)
                
                prev_acts_feed = prev_acts_feed.reshape(1, PADDING_SIZE, 1)
                
                if expertRNN_prev_obs is None:
                    expertRNN_prev_obs = prev_obs_feed
                    expertRNN_prev_acts = prev_acts_feed
                    expertRNN_tar_ob = trajs[i][0][ii+1]
                    # expertRNN_tar_act = actions[-1]
                else:
                    expertRNN_prev_obs = np.concatenate((expertRNN_prev_obs, prev_obs_feed), axis=0)
                    expertRNN_prev_acts = np.concatenate((expertRNN_prev_acts, prev_acts_feed), axis=0)
                    expertRNN_tar_ob = np.concatenate((expertRNN_tar_ob, trajs[i][0][ii+1]), axis=0)
                    # expertRNN_tar_act = np.concatenate((expertRNN_tar_act, actions[-1]), axis=0)
                expertRNN_tar_act.append(trajs[i][1][ii+1])
            
            # ===================================================
            # # Use whole expert episode as input for expert RNN
            # # for exp RNN training
            # prev_obs_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(trajs[i][0][:-1]).flatten()], maxlen=ob_space.shape[0]*PADDING_SIZE, padding='post')
            # # print(prev_obs_feed.shape)
            
            # prev_acts_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(trajs[i][1][:-1]).flatten()], maxlen=PADDING_SIZE)

            # prev_obs_feed = prev_obs_feed.reshape(1, PADDING_SIZE, ob_space.shape[0])
            # # print(prev_obs_feed.shape)
            
            # prev_acts_feed = prev_acts_feed.reshape(1, PADDING_SIZE, 1)
            
            # if expertRNN_prev_obs is None:
                # expertRNN_prev_obs = prev_obs_feed
                # expertRNN_prev_acts = prev_acts_feed
                # expertRNN_tar_ob = trajs[i][0][-1]
                # # expertRNN_tar_act = actions[-1]
            # else:
                # expertRNN_prev_obs = np.concatenate((expertRNN_prev_obs, prev_obs_feed), axis=0)
                # expertRNN_prev_acts = np.concatenate((expertRNN_prev_acts, prev_acts_feed), axis=0)
                # expertRNN_tar_ob = np.concatenate((expertRNN_tar_ob, trajs[i][0][-1]), axis=0)
                # # expertRNN_tar_act = np.concatenate((expertRNN_tar_act, actions[-1]), axis=0)
            # expertRNN_tar_act.append(trajs[i][1][-1])
            
            # # Use whole expert episode as input for expert RNN
            # ===================================================
            
            # expertRNN_prev_obs.append(prev_obs_feed)
            # expertRNN_prev_acts.append(prev_acts_feed)
            # expertRNN_tar_ob.append(observations[-1])
            # expertRNN_tar_act.append(actions[-1])
                
        
        expertRNN_tar_ob = np.reshape(expertRNN_tar_ob, newshape=[-1] + list(ob_space.shape))
        
        expertRNN_tar_act = np.array(expertRNN_tar_act)
        expertRNN_tar_act = np.reshape(expertRNN_tar_act, newshape=[-1, 1])
        
        # expertRNN_prev_obs = np.array(expertRNN_prev_obs)
        # expertRNN_prev_acts = np.array(expertRNN_prev_acts)
        # expertRNN_tar_ob = np.array(expertRNN_tar_ob)
        # expertRNN_tar_act = np.array(expertRNN_tar_act)
        # print(expert_observations)
        # print(expert_observations.shape)
        # print(expert_actions)
        # print(expert_actions.shape)
        
        # train RNN for episode expert useage
        # for i in range(8):
            # Exp_RNN.train(prev_obs=expertRNN_prev_obs,
                    # prev_acts=expertRNN_prev_acts,
                    # tar_ob=expertRNN_tar_ob,
                    # tar_act=expertRNN_tar_act)
                    
                    
        # train RNN for s,a pair expert usage
        exprnn_inp = [expertRNN_prev_obs, expertRNN_prev_acts, expertRNN_tar_ob, expertRNN_tar_act]
        for epoch in range(8):
            sample_indices = np.random.randint(low=0, high=expertRNN_prev_obs.shape[0],
                                                size=32)  # indices are in [low, high)
            sampled_exprnn_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in exprnn_inp]  # sample training data
            Exp_RNN.train(prev_obs=sampled_exprnn_inp[0],
                    prev_acts=sampled_exprnn_inp[1],
                    tar_ob=sampled_exprnn_inp[2],
                    tar_act=sampled_exprnn_inp[3])


        # ====================================================
        # Negative RNN
        # ====================================================
        
        negativeRNN_prev_obs = None
        negativeRNN_prev_acts = None
        negativeRNN_tar_ob = None
        negativeRNN_tar_act = []
        iteration = len(running_paths)
        number_negative_sample = math.ceil(NEGATIVE_RATIO*iteration)
        for i in traj_order[-number_negative_sample:]:
            
            # ===================================================
            # # Use every s,a pair as input to train expert rnn
            # for ii in range(len(trajs[i][0]) - 1):
                # prev_obs_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(trajs[i][0][:ii+1]).flatten()], maxlen=ob_space.shape[0]*PADDING_SIZE, padding='post')
                # # print(prev_obs_feed.shape)
                
                # prev_acts_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(trajs[i][1][:ii+1]).flatten()], maxlen=PADDING_SIZE)

                # prev_obs_feed = prev_obs_feed.reshape(1, PADDING_SIZE, ob_space.shape[0])
                # # print(prev_obs_feed.shape)
                
                # prev_acts_feed = prev_acts_feed.reshape(1, PADDING_SIZE, 1)
                
                # if negativeRNN_prev_obs is None:
                    # negativeRNN_prev_obs = prev_obs_feed
                    # negativeRNN_prev_acts = prev_acts_feed
                    # negativeRNN_tar_ob = trajs[i][0][ii+1]
                    # # negativeRNN_tar_act = actions[-1]
                # else:
                    # negativeRNN_prev_obs = np.concatenate((negativeRNN_prev_obs, prev_obs_feed), axis=0)
                    # negativeRNN_prev_acts = np.concatenate((negativeRNN_prev_acts, prev_acts_feed), axis=0)
                    # negativeRNN_tar_ob = np.concatenate((negativeRNN_tar_ob, trajs[i][0][ii+1]), axis=0)
                    # # negativeRNN_tar_act = np.concatenate((expertRNN_tar_act, actions[-1]), axis=0)
                # negativeRNN_tar_act.append(trajs[i][1][ii+1])
            
            # ===================================================
            # Use whole negative episode as input for negative RNN
            # for neg RNN training
            prev_obs_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(trajs[i][0][:-1]).flatten()], maxlen=ob_space.shape[0]*PADDING_SIZE, padding='post')
            # print(prev_obs_feed.shape)
            
            prev_acts_feed = tf.keras.preprocessing.sequence.pad_sequences([np.array(trajs[i][1][:-1]).flatten()], maxlen=PADDING_SIZE)

            prev_obs_feed = prev_obs_feed.reshape(1, PADDING_SIZE, ob_space.shape[0])
            # print(prev_obs_feed.shape)
            
            prev_acts_feed = prev_acts_feed.reshape(1, PADDING_SIZE, 1)
            
            if negativeRNN_prev_obs is None:
                negativeRNN_prev_obs = prev_obs_feed
                negativeRNN_prev_acts = prev_acts_feed
                negativeRNN_tar_ob = trajs[i][0][-1]
                # negativeRNN_tar_act = actions[-1]
            else:
                negativeRNN_prev_obs = np.concatenate((negativeRNN_prev_obs, prev_obs_feed), axis=0)
                negativeRNN_prev_acts = np.concatenate((negativeRNN_prev_acts, prev_acts_feed), axis=0)
                negativeRNN_tar_ob = np.concatenate((negativeRNN_tar_ob, trajs[i][0][-1]), axis=0)
                # negativeRNN_tar_act = np.concatenate((negativeRNN_tar_act, actions[-1]), axis=0)
            negativeRNN_tar_act.append(trajs[i][1][-1])
            
            # Use whole expert episode as input for expert RNN
            # ===================================================
            
            # expertRNN_prev_obs.append(prev_obs_feed)
            # expertRNN_prev_acts.append(prev_acts_feed)
            # expertRNN_tar_ob.append(observations[-1])
            # expertRNN_tar_act.append(actions[-1])
        
        negativeRNN_tar_ob = np.reshape(negativeRNN_tar_ob, newshape=[-1] + list(ob_space.shape))
        
        negativeRNN_tar_act = np.array(negativeRNN_tar_act)
        negativeRNN_tar_act = np.reshape(negativeRNN_tar_act, newshape=[-1, 1])
        
        # expertRNN_prev_obs = np.array(expertRNN_prev_obs)
        # expertRNN_prev_acts = np.array(expertRNN_prev_acts)
        # expertRNN_tar_ob = np.array(expertRNN_tar_ob)
        # expertRNN_tar_act = np.array(expertRNN_tar_act)
        # print(expert_observations)
        # print(expert_observations.shape)
        # print(expert_actions)
        # print(expert_actions.shape)
        
        # train RNN for episode expert useage
        # for i in range(8):
            # Neg_RNN.train(prev_obs=negativeRNN_prev_obs,
                    # prev_acts=negativeRNN_prev_acts,
                    # tar_ob=negativeRNN_tar_ob,
                    # tar_act=negativeRNN_tar_act)
                    
                    
        # train RNN for s,a pair expert usage
        negrnn_inp = [negativeRNN_prev_obs, negativeRNN_prev_acts, negativeRNN_tar_ob, negativeRNN_tar_act]
        for epoch in range(8):
            sample_indices = np.random.randint(low=0, high=negativeRNN_prev_obs.shape[0],
                                                size=16)  # indices are in [low, high)
            sampled_negrnn_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in negrnn_inp]  # sample training data
            Neg_RNN.train(prev_obs=sampled_negrnn_inp[0],
                    prev_acts=sampled_negrnn_inp[1],
                    tar_ob=sampled_negrnn_inp[2],
                    tar_act=sampled_negrnn_inp[3])
        
        # ================================================

        pol_obs, pol_acs, atarg, atarg_ER, tdlamret, tdlamret_ER, pol_prev_obs, pol_prev_acs, exp_actions_prob, neg_actions_prob = rollout["obs"], rollout["acs"],  rollout["adv"], rollout["adv_ER"], rollout["tdlamret"], rollout["tdlamret_ER"], rollout["prev_obs"], rollout["prev_acs"],  Exp_RNN.get_action_prob(rollout["prev_obs"], rollout["prev_acs"]), Neg_RNN.get_action_prob(rollout["prev_obs"], rollout["prev_acs"])

        # Standardized advantage function estimate
        atarg = (atarg - atarg.mean()) / atarg.std()
        atarg_ER = (atarg_ER - atarg_ER.mean()) / atarg_ER.std()

        d = dataset.Dataset(dict(obs=pol_obs, acs=pol_acs, atarg=atarg, atarg_ER=atarg_ER, vtarg=tdlamret,
            vtarg_ER=tdlamret_ER, oldvpred=rollout["vpreds"], oldvpred_ER=rollout["vpreds_ER"], prev_obs=pol_prev_obs, prev_acs=pol_prev_acs, exp_actions_prob=exp_actions_prob, neg_actions_prob=neg_actions_prob), deterministic=pi.recurrent)

        # update running mean/std for policy
        if hasattr(pi, "ob_rms"): pi.ob_rms.update(pol_obs)

        assign_old_eq_new() # set old parameter values to new parameter values
        logger.log("Optimizing...")
        logger.log(fmt_row(13, loss_names))

        # PPO optimization
        for _ in range(optim_epochs):
            losses = [] # list of tuples, each of which gives the loss for a minibatch
            for batch in d.iterate_once(optim_batchsize):

                # "g" is policy gradient using discriminator rewards, while
                # "g_ER" is policy gradient using environmental rewards
                *pi_newlosses, g = pi_lossandgrad(batch["obs"], batch["acs"], batch["atarg"], cur_lrmult, batch["exp_actions_prob"], batch["neg_actions_prob"])
                *pi_newlosses_ER, g_ER = pi_lossandgrad(batch["obs"], batch["acs"], batch["atarg_ER"], cur_lrmult, batch["exp_actions_prob"], batch["neg_actions_prob"])
                pi_newlosses_avg = np.average(np.array([pi_newlosses, pi_newlosses_ER]), axis=0, weights=[mu, 1-mu])

                # Update policy with averaged gradient
                g_avg = np.average(np.array([g, g_ER]), axis=0, weights=[mu, 1-mu])
                pi_adam.update(g_avg, optim_stepsize * cur_lrmult)

                # Update both critics
                *vf_newlosses, g_vf = vf_lossandgrad(batch["obs"], batch["vtarg"], batch["vtarg_ER"], batch["oldvpred"], batch["oldvpred_ER"], cur_lrmult)
                vf_adam.update(g_vf, optim_stepsize * cur_lrmult)

                losses.append(list(pi_newlosses_avg) + vf_newlosses)

            logger.log(fmt_row(13, np.mean(losses, axis=0)))

        logger.log("Evaluating losses...")
        losses = []
        for batch in d.iterate_once(optim_batchsize):
            pi_newlosses = compute_pi_losses(batch["obs"], batch["acs"], batch["atarg"], cur_lrmult, batch["exp_actions_prob"], batch["neg_actions_prob"])
            pi_newlosses_ER = compute_pi_losses(batch["obs"], batch["acs"], batch["atarg_ER"], cur_lrmult, batch["exp_actions_prob"], batch["neg_actions_prob"])
            pi_newlosses_avg = np.average(np.array([pi_newlosses, pi_newlosses_ER]), axis=0, weights=[mu, 1-mu])
            vf_newlosses = compute_vf_losses(batch["obs"], batch["vtarg"], batch["vtarg_ER"], batch["oldvpred"], batch["oldvpred_ER"], cur_lrmult)
            losses.append(list(pi_newlosses_avg) + vf_newlosses)

        meanlosses = np.mean(losses, axis=0)
        logger.log(fmt_row(13, meanlosses))
        for (lossval, name) in zipsame(meanlosses, loss_names):
            logger.record_tabular("loss_"+name, lossval)

        logger.record_tabular("ev_tdlam_before", explained_variance(rollout["vpreds"], tdlamret))
        logger.record_tabular("ev_tdlam_before_ER", explained_variance(rollout["vpreds_ER"], tdlamret_ER))

        lens, rets_oracle = rollout["ep_lens"], rollout["ep_rets_oracle"]
        log_record.insert(key='environment_episodic_returns', value=rets_oracle)
        rets_oracle_buffer.extend(rets_oracle)
        lenbuffer.extend(lens)

        logger.record_tabular("EpLenMean", np.mean(lenbuffer))
        logger.record_tabular("EpOracleRetMean", np.mean(rets_oracle_buffer))
        logger.record_tabular("EpThisIter", len(lens))
        episodes_so_far += len(lens)
        timesteps_so_far += timesteps_per_batch

        logger.record_tabular("EpisodesSoFar", episodes_so_far)
        logger.record_tabular("TimestepsSoFar", timesteps_so_far)
        logger.record_tabular("TimeElapsed", time.time() - tstart)
        logger.dump_tabular()

        # ------------------ Update D ------------------
        logger.log("Optimizing Discriminator...")
        logger.log(fmt_row(13, discriminator.loss_names))

        d_losses = [] # list of tuples, each of which gives the loss for a d_step
        # d_step updates for the discriminator
        for pol_data, expert_data in zip(
                dataset.iterbatches((pol_obs, pol_acs), batch_size=(timesteps_per_batch // d_step)),
                dataset.iterbatches((pq_replay.obs, pq_replay.acs, pq_replay.wts), batch_size=(len(pq_replay) // d_step))):

            pol_obs_mb, pol_acs_mb = pol_data
            expert_obs_mb, expert_acs_mb, expert_wts_mb = expert_data
            expert_wts_mb /= np.sum(expert_wts_mb)

            # print(pol_obs_mb.shape, expert_obs_mb.shape)
            # print(pol_acs_mb.shape, expert_acs_mb.shape)
            # print(np.concatenate((pol_obs_mb, expert_obs_mb), axis=0).shape)
            # print(np.concatenate((pol_acs_mb, expert_acs_mb), axis=0).shape)
            # print(np.expand_dims(np.concatenate((pol_acs_mb, expert_acs_mb), axis=0), axis=1).shape)
            
            # update input (ob+ac) mean/std for discriminator
            if hasattr(discriminator, "in_rms"): discriminator.in_rms.update(
                    np.concatenate((
                        np.concatenate((pol_obs_mb, expert_obs_mb), axis=0),
                        np.expand_dims(np.concatenate((pol_acs_mb, expert_acs_mb), axis=0), axis=1)),
                        axis=1))

            *newlosses, g = discriminator.lossandgrad(pol_obs_mb, pol_acs_mb, expert_obs_mb, expert_acs_mb, expert_wts_mb)
            d_adam.update(g, d_stepsize * cur_lrmult)
            d_losses.append(newlosses)

        logger.log(fmt_row(13, np.mean(d_losses, axis=0)))

        iters_so_far += 1
        
        if iteration_num <= episodes_so_far:
            break

    log_record.dump()
    resultf.close()
