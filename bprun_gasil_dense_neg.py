#!/usr/bin/python3
import argparse
import gym
import numpy as np
import tensorflow as tf
from network_models.policy_net import Policy_net
from network_models.discriminator import Discriminator
from network_models.negative_policy_nopsv import Negative_policy
from algo.ppo_neg import PPOTrain
from tqdm import trange

from env.epidemic import EpidemicEnvDecay
# from env.epidemic import EpidemicEnv
import pickle
from prun_util import calc_prob
import networkx as nx

from scipy.stats import entropy

import random

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

PADDING_SIZE = 20
EXPERT_SIZE = 20

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/epidemicdecay/gasil')
    parser.add_argument('--savedir', help='save directory', default='trained_models/epidemicdecay/gasil')
    parser.add_argument('--resultdir', help='result directory', default='result/epidemicdecay/gasil')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e3))
    parser.add_argument('--graphpath', help='graph path', default='env/testG.pkl')
    parser.add_argument('--starttime', default=int(5), type=int)
    parser.add_argument('--timestep', default=int(1))
    parser.add_argument('--stagenumber', default=int(20), type=int)
    parser.add_argument('--endtime', default=int(30))
    parser.add_argument('--budget', default=int(20))
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    return parser.parse_args()


def main(args):
    #env = gym.make('CartPole-v0')
    fo = open(args.graphpath, "rb")
    N, G, H, J, IC, return_statuses, user_cost = pickle.load(fo)
    
    
    total_budget = float(args.budget)
    
    J = nx.DiGraph()
    J.add_edge(('I', 'S'), ('I', 'EI'), rate = 1.0, rate_function=calc_prob)
    J.add_edge(('R', 'S'), ('R', 'ER'), rate = 1.0, rate_function=calc_prob)
    J.add_edge(('I', 'R'), ('I', 'I'), rate = 1.0, rate_function=calc_prob)
    J.add_edge(('R', 'I'), ('R', 'R'), rate = 1.0, rate_function=calc_prob)
    
    env = EpidemicEnvDecay(N, G, H, J, IC, return_statuses, args.starttime, float(args.timestep), args.stagenumber, int(float(args.endtime)))
    
    env.seed(args.seed)

    # add fix seed 
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    ob_space = env.observation_space
    Policy = Policy_net('policy', env)
    Old_Policy = Policy_net('old_policy', env)
    # Exp_RNN = SARNN('expert_RNN', env)
    Neg_Policy = Negative_policy("neg_policy", env)
    PPO = PPOTrain(Policy, Old_Policy, gamma=args.gamma, c_3=1.0)
    D = Discriminator(env)

    print(env.observation_space)
    print(env.action_space)
    print(env.observation_space.shape)
    # print(env.observation_space.low)
    # print(env.observation_space.high)
    # expert_observations = np.genfromtxt('trajectory/observations.csv')
    # expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)
    
    # print(expert_observations)
    # print(expert_observations.shape)
    # print(expert_actions)
    # print(expert_actions.shape)
    resultf = open(args.resultdir, "w")
    saver = tf.train.Saver()

    with tf.Session() as sess:
        writer = tf.summary.FileWriter(args.logdir, sess.graph)
        sess.run(tf.global_variables_initializer())

        obs = env.reset()
        success_num = 0
        # avoid pointer
        running_budget = 0 + total_budget

        action_counts = np.zeros((env.action_space.n))

        trajs = []
        traj_rewards = []

        for iteration in trange(int(args.iteration)):
            observations = []
            step_observations = [obs]
            actions = []
            # do NOT use rewards to update policy
            rewards = []
            v_preds = []
            run_policy_steps = 0
            while True:
                run_policy_steps += 1

                obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                act, v_pred = Policy.act(obs=obs, stochastic=True)

                infected_user = obs[0][N : 2*N]
                recovered_user = obs[0][3*N : 4*N]
                infected_prop_num = obs[0][0 : N]
                recovered_prop_num = obs[0][2*N : 3*N]
                num_followers = obs[0][4*N : 5*N]
                
                user_under_budget = np.zeros(N)
                for i in range(N):
                    if running_budget >= user_cost[i]:
                        user_under_budget[i] = 1
                
                # act, v_pred = Policy.act(obs=obs, stochastic=True)
                # print(act)
                # print(len(act))
                # print(len(set(act)))
                action_space = range(N)
                picked_act = None
                for act_indicator in act:
                    # state of user not recovered and under our budget
                    if user_under_budget[act_indicator] == 1:
                        picked_act = act_indicator
                        break
                if picked_act is None:
                    # pick randomly from selectable users
                    picked_act = random.choices(action_space, weights = user_under_budget, k = 1)[0]

                action_counts[picked_act] += 1
                act = np.asscalar(np.array([picked_act]))
                
                v_pred = np.asscalar(v_pred)
                
                toEnd = True
                left_budget = running_budget - user_cost[picked_act]
                for i in range(N):
                    # if have users still selectable
                    if left_budget >= user_cost[i]:
                        toEnd = False
                        break
                
                next_obs, reward, done, info = env.step(act, toEnd = toEnd)
                
                running_budget = running_budget - user_cost[picked_act]
                
                step_observations.append(next_obs)
                observations.append(obs)
                actions.append(act)
                rewards.append(reward)
                v_preds.append(v_pred)

                if done:
                    next_obs = np.stack([next_obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
                    _, v_pred = Policy.act(obs=next_obs, stochastic=True)
                    v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                    obs = env.reset()
                    
                    # reset running budget
                    running_budget = 0 + total_budget
                    
                    break
                else:
                    obs = next_obs

            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_length', simple_value=run_policy_steps)])
                               , iteration)
            writer.add_summary(tf.Summary(value=[tf.Summary.Value(tag='episode_reward', simple_value=sum(rewards))])
                               , iteration)
            resultf.write(str(sum(rewards)) + "\n")
            
            print("ep_reward:", str(sum(rewards)))
            # if sum(rewards) >= 195:
                # success_num += 1
                # if success_num >= 100:
                    # saver.save(sess, args.savedir + '/model.ckpt')
                    # print('Clear!! Model saved.')
                    # break
            # else:
                # success_num = 0
                
                
            print(min(list(user_cost.values())))
            print("Done ",len(actions), " actions.")
            print("Population entropy:", entropy(action_counts))
            
            # pick expert obs and actions
            trajs.append([observations, actions])
            traj_rewards.append(sum(rewards))
            traj_order = np.argsort(traj_rewards)[::-1]
            # print(trajs)
            # print(traj_rewards)
            # print(traj_order)
            expert_observations = []
            expert_actions = []
            
            expertRNN_prev_obs = None
            expertRNN_prev_acts = None
            expertRNN_tar_ob = None
            expertRNN_tar_act = []
            for i in traj_order[:EXPERT_SIZE]:
            # for i in random.choices(traj_order[:20], k = 10):
                # for D training
                expert_observations.extend(trajs[i][0])
                expert_actions.extend(trajs[i][1])
                
                    
            expert_observations = np.array(expert_observations)
            expert_observations = np.reshape(expert_observations, newshape=[-1, ob_space.shape[0]])
            expert_actions = np.array(expert_actions)
            

            negative_observations = []
            negative_actions = []

            # fast fix
            NEGATIVE_RATIO = 0.1
            import math

            if iteration > 40:
            # if iteration > 40 and iteration % 20 ==0: 
                # set mininal size as EXPERT_SIZE
                number_negative_sample = math.ceil(NEGATIVE_RATIO*iteration)
                # if number_negative_sample < EXPERT_SIZE:
                #     number_negative_sample = EXPERT_SIZE
                print("Expert top reward:", traj_rewards[traj_order[EXPERT_SIZE]])
                print("Negative top ratio reward:", traj_rewards[traj_order[-number_negative_sample]])
                print("Negative top expertnum reward:", traj_rewards[traj_order[-EXPERT_SIZE]])
                for i in traj_order[-EXPERT_SIZE:]:
                # for i in traj_order[-number_negative_sample:]:
                # for i in traj_order[-EXPERT_SIZE:]:
                # for i in random.choices(traj_order[:20], k = 10):
                    
                    # pick all traj as input
                    negative_observations.extend(trajs[i][0])
                    negative_actions.extend(trajs[i][1])

                    # pick last 5 as input
                    # if len(trajs[i][0]) > 5:
                    #     negative_observations.extend(trajs[i][0][-5:])
                    #     negative_actions.extend(trajs[i][1][-5:])
                    # else:
                    #     negative_observations.extend(trajs[i][0])
                    #     negative_actions.extend(trajs[i][1])

                    # skip first one
                    # negative_observations.extend(trajs[i][0][1:])
                    # negative_actions.extend(trajs[i][1][1:])

                    # pick last one as input
                    # negative_observations.extend(trajs[i][0][-1:])
                    # negative_actions.extend(trajs[i][1][-1:])


                negative_observations = np.array(negative_observations)
                negative_observations = np.reshape(negative_observations, newshape=[-1, ob_space.shape[0]])
                negative_actions = np.array(negative_actions)
                negative_actions = np.reshape(negative_actions, newshape=[-1, 1]) 
                
                # print(negative_observations.shape)
                # print(negative_actions.shape, negative_actions)
                # print(negative_actions[0])


                # train RNN for s,a pair expert usage
                # negrnn_inp = [negative_observations, negative_actions]

                # for epoch in range(32):
                #     sample_indices = np.random.randint(low=0, high=negative_observations.shape[0],
                #                                     size=64)  # indices are in [low, high)
                #     sampled_negrnn_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in negrnn_inp]  # sample training data
                #     # print(sampled_negrnn_inp[0].shape, sampled_negrnn_inp[1].shape)

                #     Neg_Policy.train(obs=sampled_negrnn_inp[0],
                #             acts=sampled_negrnn_inp[1])
            
                # Full data as input
                for i in range(32):
                    Neg_Policy.train(obs=negative_observations,
                            acts=negative_actions)


            # convert list to numpy array for feeding tf.placeholder
            observations = np.reshape(observations, newshape=[-1, ob_space.shape[0]] )
            actions = np.array(actions).astype(dtype=np.int32)
            

            # train discriminator
            if iteration not in traj_order[:EXPERT_SIZE]:
                for i in range(1):
                    D.train(expert_s=expert_observations,
                            expert_a=expert_actions,
                            agent_s=observations,
                            agent_a=actions)

            # output of this discriminator is reward
            d_rewards = D.get_rewards(agent_s=observations, agent_a=actions)
            d_rewards = np.reshape(d_rewards, newshape=[-1]).astype(dtype=np.float32)

            d_neg_actions = Neg_Policy.get_action_prob(obs=observations)

            gaes = PPO.get_gaes(rewards=d_rewards, v_preds=v_preds, v_preds_next=v_preds_next)
            gaes = np.array(gaes).astype(dtype=np.float32)
            # gaes = (gaes - gaes.mean()) / gaes.std()
            v_preds_next = np.array(v_preds_next).astype(dtype=np.float32)
            
            # print(exp_actions.shape)
            # train policy
            # Skip the first one s,a pair
            inp = [observations, actions, gaes, d_rewards, v_preds_next, d_neg_actions]
            # inp = [observations[1:], actions[1:], gaes[1:], d_rewards[1:], v_preds_next[1:], exp_actions]
            if observations.shape[0] > 1:
                PPO.assign_policy_parameters()
                for epoch in range(8):
                    sample_indices = np.random.randint(low=0, high=observations[1:].shape[0],
                                                    size=16)  # indices are in [low, high)
                    sampled_inp = [np.take(a=a, indices=sample_indices, axis=0) for a in inp]  # sample training data
                    PPO.train(obs=sampled_inp[0],
                            actions=sampled_inp[1],
                            gaes=sampled_inp[2],
                            rewards=sampled_inp[3],
                            v_preds_next=sampled_inp[4],
                            neg_act_probs=sampled_inp[5])

                summary = PPO.get_summary(obs=inp[0],
                                        actions=inp[1],
                                        gaes=inp[2],
                                        rewards=inp[3],
                                        v_preds_next=inp[4],
                                        neg_act_probs=inp[5])
                                        
                summary_proto = tf.Summary().FromString(summary)
                print("PPO SUMMARY:", summary_proto)
                
                writer.add_summary(summary, iteration)
        writer.close()
        resultf.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
