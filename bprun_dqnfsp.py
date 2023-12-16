# -*- coding: utf-8 -*-
import argparse
import random
#import gym
import numpy as np
from collections import deque
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
# from hawkes_env import HawkesEnv
import sys
import tensorflow as tf
import tensorflow.keras.backend as K

# using the BRAND NEW api
# from tensorflow.python.keras.callbacks import TensorBoard

from env.epidemic import EpidemicEnvDecay
# from env.epidemic import EpidemicEnv
import pickle
from prun_util import calc_prob
import networkx as nx

from tensorflow.core.protobuf import rewriter_config_pb2

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
off = rewriter_config_pb2.RewriterConfig.OFF
config.graph_options.rewrite_options.arithmetic_optimization = off
config.graph_options.rewrite_options.memory_optimization  = off
sess = tf.Session(config=config)
K.set_session(sess)

np.random.seed(0) 


# EPISODES = 900
# TEST_EPISODES = 100
# reward_type 0 => average, 1 => reward by predicted state, 2 => reward by dqn one action reward, 3 => reward by gibbs sampling
REWARD_TYPE = 1

def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', help='log directory', default='log/epidemicdecay/dqnfsp')
    parser.add_argument('--savedir', help='save directory', default='trained_models/epidemicdecay/dqnfsp')
    parser.add_argument('--resultdir', help='result directory', default='result/epidemicdecay/dqnfsp')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e3))
    parser.add_argument('--graphpath', help='graph path', default='env/bG_500_08')
    parser.add_argument('--starttime', default=int(5), type=int)
    parser.add_argument('--timestep', default=int(1))
    parser.add_argument('--stagenumber', default=int(20), type=int)
    parser.add_argument('--endtime', default=int(30))
    parser.add_argument('--budget', default=int(20))
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    return parser.parse_args()
    

class DQNAgent:
    def __init__(self, state_size, action_size, logdir = "./tbsession"):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=200)
        self.gamma = 0.95    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        # self.tbCallBack = TensorBoard(log_dir=logdir)
        self.epochcount = 1

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mae',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    # action_cost is a list of cost corrsponding to every action
    def act(self, state, action_cost, budget, blocked_actions):

        if np.random.rand() <= self.epsilon:
            avaliable_actions = []
            for i in range(len(action_cost)):
                if i not in blocked_actions and budget >= action_cost[i]:
                    avaliable_actions.append(i)
            #return random.randrange(self.action_size)
            return random.choice(avaliable_actions)
        
        act_values = self.model.predict(state)
        mask_count = 0
        # print(act_values.shape)
        # print(act_values)
        # print(action_cost)
        for i in range(len(action_cost)):
            if budget < action_cost[i] or i in blocked_actions:
                # set this action will not be selected
                act_values[0][i] = 0
                mask_count += 1
        # print("masked " + str(mask_count))

        
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        states, targets_f = [], []
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target 
            # Filtering out states and targets for training
            states.append(state[0])
            targets_f.append(target_f[0])
        # history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        history = self.model.fit(np.array(states), np.array(targets_f), epochs=self.epochcount+1, initial_epoch=self.epochcount, verbose=0)
        self.epochcount += 1
        # Keeping track of loss
        loss = history.history['loss'][0]
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)



class FSP:
    def __init__(self, state_size):

        self.model = self.build_model(state_size)
        self.memory = deque(maxlen=200)

    def memorize(self, init_state, actions, final_state):
        self.memory.append((init_state, actions, final_state))

    def clearmem(self):
        self.memory.clear()


    def build_model(self, state_size):
        model = Sequential()
        
        # input have variate time stamps with one action
        # model.add(Dense(128, input_dim=state_size, batch_input_shape=(1, None, 1)))
        model.add(LSTM(state_size, 
                    batch_input_shape=(1, None, 1),
                    activation='relu',
                    stateful=True))


        # model.add(Dropout(0.5))
        model.compile(loss='mae', optimizer='adam', metrics=['mae'])
        model.summary()
        return model

    def replay(self, batch_size):
        # for n in range(16):
        minibatch = random.sample(self.memory, batch_size)
        #actions, final_states = [], []
        for init_state, action, final_state in minibatch:
            #actions.append(action[0])
            #final_states.append(final_state[0])
        # history = self.model.fit(np.array(states), np.array(targets_f), epochs=1, verbose=0)
        # history = self.model.fit(np.array(states), np.array(targets_f), epochs=self.epochcount+1, initial_epoch=self.epochcount, verbose=0, callbacks=[self.tbCallBack])
            # print("fsptrainstep")
            # print(init_state, action, final_state)
            history = self.train(init_state, action, final_state)


    def train(self, init_state, actions, final_state):
        # reset state
        self.model.layers[0].reset_states(states=[init_state, init_state])
        #print(actions)
        #print(self.model.layers[0].states)
        #hidden_states = K.variable(value=(init_state))
        #hidden_states = K.variable(value=np.random.normal(size=(1, len(init_state[0]))))
        # set hidden state
        #self.model.layers[0].states[0] = hidden_states
        # print(actions)
        # print(actions.shape)
        actions = tf.convert_to_tensor(actions, dtype=tf.int64) 
        
        self.model.fit(actions, final_state, verbose=0, steps_per_epoch=1)

        return 0
    
    def predict(self, init_state, actions):
        # reset state
        self.model.layers[0].reset_states(states=[init_state, init_state])

        #self.model.layers[0].states[0] = init_state
        actions = tf.convert_to_tensor(actions, dtype=tf.int64) 
        
        # print(actions)
        # print(actions.shape.dim)
        final_state = self.model.predict(actions, steps=1)
        # print(final_state)
        # print(final_state.shape)

        return final_state


if __name__ == "__main__":

    args = argparser()

    fo = open(args.graphpath, "rb")
    N, G, H, J, IC, return_statuses, user_cost = pickle.load(fo)
    fo.close()
    
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

    # ob_space = env.observation_space
    
    # env = HawkesEnv(n_nodes, adjacency = adjacency, decays = decays, baseline = baseline, edges = edges)
    
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    
    # print(state_size, action_size)
    
    #print(action_size)
    # for i in range(action_size):
    #     action_cost.append(random.normalvariate(0.1, 0.1))
    # budget = []
    # for i in range(500):
    #     budget.append(random.normalvariate(0.1, 0.1))
    agent = DQNAgent(state_size, action_size, args.logdir )
    # agent.load("./save/cartpole-dqn.h5")
    done = False
    batch_size = 4
    
    fsp = FSP(state_size)

    cumm_reward_hist = []
    loss_list = []
    
    resultf = open(args.resultdir, "w")
    EPISODES = int(args.iteration) - 100
    for e in range(EPISODES):
        
        print("starting episode:" + str(e))
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # env.set_fake(fake_nodes)
        episode_cumm_reward = 0
        actual_episode_time = 1
        for time in range(actual_episode_time):
            print("acting " + str(time))

            actions = np.zeros(N)
            blocked_actions = []
            action_list = []
            # time_budget = budget[time]
            time_budget = total_budget
            action_cost = user_cost
            #action_cost = np.zeros(N)
            # set current state
            curr_state = state
            curr_state = np.reshape(curr_state, [1, state_size])
            
            infected_user = curr_state[0][N : 2*N]
            recovered_user = curr_state[0][3*N : 4*N]
            infected_recv_num = curr_state[0][0 : N]
            recovered_recv_num = curr_state[0][2*N : 3*N]
            
            for act_indicator in range(len(recovered_user)):
                if recovered_user[act_indicator] == 1:
                    blocked_actions.append(act_indicator)
            # block fake news spreader
            # for f in fake_nodes_choice:
                # blocked_actions.append(f)

            # control the selection of actions

            # budget_left = False
            # for i in set(range(n_nodes)).difference(set(blocked_actions)):
                # if action_cost[i] < time_budget:
                    # budget_left = True
            
            # if budget_left == True:
            action = agent.act(state, action_cost, time_budget, blocked_actions)
            actions[action] = 1
            action_list.append(action)
            # cannot select same node twice
            blocked_actions.append(action)
            # time_budget = time_budget - action_cost[action]
                
            # for i in range(20):
            #     actions[i] = 1
            #print(actions, int(actual_episode_time[time]))
            next_state, epreward, isEnd, _, reward = env.step(action, withreward=True)
            
            # next_state, reward = env.step(actions, int(actual_episode_time[time]), mitigate_intensity)
            episode_cumm_reward += epreward

            # print(len(action_list))
            print(reward)
            # reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            
            
            # remember state change
            fsp.memorize(np.reshape(curr_state, [1, state_size]), np.reshape(np.array(action_list), [1, len(action_list), 1]), next_state)
            
            # average reward of every action
            for act in action_list:
                # act_reward = reward / len(action_list)
                act_reward = reward
                agent.memorize(state, act, act_reward, next_state, done)

            state = next_state

            #if len(fsp.memory) > batch_size:
            #    fsp.replay(batch_size)
            if len(agent.memory) > batch_size:
                loss = agent.replay(batch_size)
                loss_list.append(loss)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                        .format(e, EPISODES, time, loss))  
        if len(fsp.memory) > batch_size:
            fsp.replay(batch_size)
        print("episode: {}/{}, e: {:.2}"
          .format(e, EPISODES, agent.epsilon))
        
        cumm_reward_hist.append(episode_cumm_reward)
        # if e % 10 == 0:
        #     agent.save("./save/hawkes-dqn.h5")

    fsp.clearmem()

    # Start testing, no train at this time
    TEST_EPISODES = 100
    for e in range(TEST_EPISODES):
        
        print("starting testing episode:" + str(e))
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        # env.set_fake(fake_nodes)
        episode_cumm_reward = 0
        actual_episode_time = 1
        for time in range(actual_episode_time):
            print("acting " + str(time))

            actions = np.zeros(N)
            action_cost = user_cost
            blocked_actions = []
            action_list = []
            time_budget = total_budget
            curr_state = state
            curr_state = np.reshape(curr_state, [1, state_size])
            # predicted_intensity = []

            infected_user = curr_state[0][N : 2*N]
            recovered_user = curr_state[0][3*N : 4*N]
            infected_recv_num = curr_state[0][0 : N]
            recovered_recv_num = curr_state[0][2*N : 3*N]


            for act_indicator in range(len(recovered_user)):
                if recovered_user[act_indicator] == 1:
                    blocked_actions.append(act_indicator)

            # control the selection of actions
            toEnd = False
            while not toEnd:
                
                action = agent.act(np.reshape(curr_state, [1, state_size]), action_cost, time_budget, blocked_actions)
                
                actions[action] = 1
                action_list.append(action)

                # print(np.reshape(np.array(action_list), [1, len(action_list), 1]))
                # print(np.reshape(np.array(action_list), [1, len(action_list), 1]).shape)
                # print(curr_state)
                # print(curr_state.shape)
                curr_state = fsp.predict(np.reshape(curr_state, [1, state_size]), np.reshape(np.array(action_list), [1, len(action_list), 1]) )[0]
                # cut data from state
                # x_m = curr_state[n_nodes * 1 : n_nodes * 1 + n_nodes]
                # x_f = curr_state[n_nodes * 3 : n_nodes * 3 + n_nodes]
                # predicted_intensity.append(np.dot(x_m, x_f.T))

                # print(curr_state)
                # print(curr_state.shape)

                # cannot select same node twice
                blocked_actions.append(action)
                time_budget = time_budget - action_cost[action]

                print("time_budget", time_budget)

                toEnd = True
                left_budget = time_budget
                for i in range(N):
                    # if have users still selectable
                    if left_budget >= user_cost[i]:
                        toEnd = False
                        break



            # for i in range(20):
            #     actions[i] = 1
            #print(actions, int(actual_episode_time[time]))

            ##### Original implementation            
            # next_state, reward, isEnd, _ = env.steps(action_list)
            # episode_cumm_reward += reward

            # # print(len(action_list))
            # print(reward)
            # # reward = reward if not done else -10
            # next_state = np.reshape(next_state, [1, state_size])
            # # curr_state = np.reshape(curr_state, [1, state_size])

            # #fsp.train(curr_state, np.reshape(np.array(action_list), [1, len(action_list), 1]), next_state)
            # fsp.memorize(np.reshape(curr_state, [1, state_size]), np.reshape(np.array(action_list), [1, len(action_list), 1]), next_state)

            # # average reward of every action

            # state = next_state

            ##### New implementation
            ##### Select actions using DQN-FSP under budget first, then apply to environment 
            print("actions selected", action_list)
            for action in action_list:
                
                if action == action_list[-1]:
                    # last one, direct to finish episode
                    next_state, reward, isEnd, _ = env.step(action, toEnd = True)
                else:
                    next_state, reward, isEnd, _ = env.step(action, toEnd = False)
                episode_cumm_reward += reward

                next_state = np.reshape(next_state, [1, state_size])
                fsp.memorize(np.reshape(curr_state, [1, state_size]), np.reshape(np.array([action]), [1, len([action]), 1]), next_state)

                state = next_state

                if isEnd:
                    print(episode_cumm_reward)
                    break


            #if len(fsp.memory) > batch_size:
            #    fsp.replay(batch_size)
            if len(agent.memory) > batch_size:
                #loss = agent.replay(batch_size)
                #loss_list.append(loss)
                # Logging training loss every 10 timesteps
                if time % 10 == 0:
                    print("episode: {}/{}, time: {}, loss: {:.4f}"
                        .format(e, TEST_EPISODES, time, loss))  
        if len(fsp.memory) > batch_size:
            fsp.replay(batch_size)
        print("episode: {}/{}, score: {}, e: {:.2}"
          .format(e, TEST_EPISODES, time, agent.epsilon))
        
        cumm_reward_hist.append(episode_cumm_reward)
        # if e % 10 == 0:
        #     agent.save("./save/hawkes-dqn.h5")

    print(cumm_reward_hist)
    # fo = open(sys.argv[1]+".reward", "w")
    for i in cumm_reward_hist:
        resultf.write(str(i) + "\n")
    resultf.close()
