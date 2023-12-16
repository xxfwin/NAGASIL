#!/usr/bin/python3
import argparse
import gym
import numpy as np
import tensorflow as tf
#from network_models.policy_net import Policy_net
#from network_models.simple_q import SimpleQ
# from algo.ppo import PPOTrain
# from algo.ppo import PPOTrain
from tqdm import trange

# import replay memory used package
import random
from collections import deque

from env.epidemic import EpidemicEnvDecay
# from env.epidemic import EpidemicEnv
import pickle
from prun_util import calc_prob
import networkx as nx

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/epidemicdecay/maxdef')
    parser.add_argument('--savedir', help='save directory', default='trained_models/epidemicdecay/maxdef')
    parser.add_argument('--resultdir', help='result directory', default='result/epidemicdecay/maxdef')
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
    #Policy = Policy_net('policy', env)
    #Q = SimpleQ(Policy)
    
    # expert_observations = np.genfromtxt('trajectory/observations.csv')
    # expert_actions = np.genfromtxt('trajectory/actions.csv', dtype=np.int32)
    
    # print(expert_observations)
    # print(expert_observations.shape)
    # print(expert_actions)
    # print(expert_actions.shape)

    resultf = open(args.resultdir, "w")
    

    obs = env.reset()
    success_num = 0
    
    action_space = range(N)

    for iteration in trange(int(args.iteration)):
        observations = []
        actions = []
        
        # avoid pointer
        running_budget = 0 + total_budget
        
        # prev_observations = []
        # prev_actions = []
        # do NOT use rewards to update policy
        rewards = []
        v_preds = []
        run_policy_steps = 0
        while True:
            run_policy_steps += 1
            obs = np.stack([obs]).astype(dtype=np.float32)  # prepare to feed placeholder Policy.obs
            # act, v_pred = Policy.act(obs=obs, stochastic=True)
            # print(len(obs))
            # print(len(obs[0]))
            infected_user = obs[0][N : 2*N]
            recovered_user = obs[0][3*N : 4*N]
            infected_prop_num = obs[0][0 : N]
            recovered_prop_num = obs[0][2*N : 3*N]
            num_followers = obs[0][4*N : 5*N]
            
            user_under_budget = np.zeros(N)
            for i in range(N):
                if running_budget >= user_cost[i]:
                    user_under_budget[i] = 1
            
            # non_recovered_prob = (recovered_user * -1) + 1
            
            act_choice_prob = infected_user * num_followers * infected_prop_num * user_under_budget
            
            act = np.argmax(act_choice_prob)
            
            # print(act)
            
            toEnd = True
            left_budget = running_budget - user_cost[act]
            for i in range(N):
                # if have users still selectable
                if left_budget >= user_cost[i]:
                    toEnd = False
                    break
            
            next_obs, reward, done, info = env.step(act, toEnd = toEnd)
            
            running_budget = running_budget - user_cost[act]


            rewards.append(reward)

            if done:
                # next_obs = np.stack([next_obs]).astype(dtype=np.float32)
                # _, v_pred = Policy.act(obs=next_obs, stochastic=True)
                # v_preds_next = v_preds[1:] + [np.asscalar(v_pred)]
                obs = env.reset()
                # reset running budget
                running_budget = 0 + total_budget
                break
            else:
                obs = next_obs
        resultf.write(str(sum(rewards)) + "\n")



    resultf.close()


if __name__ == '__main__':
    args = argparser()
    main(args)
