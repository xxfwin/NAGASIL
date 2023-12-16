import os
import sys
import argparse
import gym

from imitation.algo import ppo_budget
from imitation.network.discriminator import Discriminator
from imitation.network.mlp_policy import MlpPolicy
from imitation.common import tf_util as U
from imitation.common.misc_util import set_global_seeds
from imitation.common.monitor import Monitor

from env.epidemic import EpidemicEnvDecay
# from env.epidemic import EpidemicEnv
import pickle
from prun_util import calc_prob
import networkx as nx


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of SIDP")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--num_cpu', help='number of cpu to used', type=int, default=1)

    # Network Configuration
    parser.add_argument('--policy_hidden_size', type=int, default=100)
    parser.add_argument('--discriminator_hidden_size', type=int, default=100)

    # Training Configuration
    parser.add_argument('--policy_entcoef', help='entropy coefficiency of policy', type=float, default=0)
    parser.add_argument('--vf_coef', help='coefficient for value function loss', type=float, default=0.5)
    parser.add_argument('--discriminator_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=1)
    # parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=2e4)

    # Self-imitation Configuration
    parser.add_argument('--mu', type=float, default=1.)
    parser.add_argument('--pq_replay_size', help='Entries in priority queue (# trajectories)', type=int, default=10)
    parser.add_argument('--episodic', help='provide reward only at the last timestep', dest='episodic', action='store_true', default=True)
    
    
    parser.add_argument('--resultdir', help='result directory', default='result/epidemicdecay/sidp')
    parser.add_argument('--gamma', default=0.95)
    parser.add_argument('--iteration', default=int(1e3))
    # parser.add_argument('--iteration', default=int(1e2))
    parser.add_argument('--graphpath', help='graph path', default='env/testG.pkl')
    parser.add_argument('--starttime', default=int(5), type=int)
    parser.add_argument('--timestep', default=int(1))
    parser.add_argument('--stagenumber', default=int(20), type=int)
    parser.add_argument('--endtime', default=int(30))
    parser.add_argument('--budget', default=int(20))
    
    
    return parser.parse_args()

def main(args):
    U.make_session(num_cpu=args.num_cpu).__enter__()
    set_global_seeds(args.seed)
    
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
    
    fo.close()
    
    num_timesteps = int(args.iteration) * args.stagenumber

    # if str(env.__class__.__name__).find('TimeLimit') >= 0:
        # from imitation.common.env_wrappers import TimeLimitMaskWrapper
        # env = TimeLimitMaskWrapper(env)

    env = Monitor(env, filename=None, allow_early_resets=True)
    env.seed(args.seed)
    env_name = "epidemic"
    data_dump_path = os.path.join(os.getcwd(), 'SIDP_temp', env_name, '_'.join(['log', str(args.seed)]))

    if args.episodic:
        from imitation.common.env_wrappers import EpisodicEnvWrapper
        env = EpisodicEnvWrapper(env)

    def policy_fn(name, ob_space, ac_space):
        return MlpPolicy(name=name, ob_space=ob_space, ac_space=ac_space,
                hid_size=policy_fn.hid_size, num_hid_layers=2)

    policy_fn.hid_size = args.policy_hidden_size
    discriminator = Discriminator(env, args.discriminator_hidden_size, args.discriminator_entcoeff)
    
    total_budget = float(args.budget)
    
    ppo_budget.learn(env, policy_fn, discriminator,
            d_step=args.d_step,
            timesteps_per_batch=100,
            clip_param=0.2, entcoef=args.policy_entcoef, vf_coef=args.vf_coef,
            optim_epochs=5, optim_stepsize=1e-5, optim_batchsize=16,
            gamma=0.95, lam=0.95, d_stepsize=1e-5,
            max_timesteps=num_timesteps,
            schedule='constant',
            data_dump_path=data_dump_path,
            mu=args.mu,
            pq_replay_size=args.pq_replay_size, resultdir=args.resultdir, total_budget=total_budget, user_cost=user_cost, iteration_num = int(args.iteration))

    env.close()

if __name__ == '__main__':
    print(sys.argv)
    args = argsparser()
    main(args)
