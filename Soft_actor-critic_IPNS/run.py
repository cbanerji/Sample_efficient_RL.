import argparse
import datetime
import gym
import numpy as np
import itertools
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple, deque
from sac import SAC #import conventional SAC
from torch.utils.tensorboard import SummaryWriter
from replay_memory import ReplayMemory #import buffer
import bonus as updt
import math
import time

device = torch.device("cpu")

parser = argparse.ArgumentParser(description='PyTorch Soft Actor-Critic Args')
parser.add_argument('--env-name', default="HalfCheetah-v2",
                    help='Mujoco Gym environment (default: HalfCheetah-v2)')
parser.add_argument('--policy', default="Gaussian",
                    help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
parser.add_argument('--eval', type=bool, default=True,
                    help='Evaluates a policy a policy every 10 episode (default: True)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for reward (default: 0.99)')
parser.add_argument('--tau', type=float, default=0.005, metavar='G',
                    help='target smoothing coefficient(τ) (default: 0.005)')
parser.add_argument('--lr', type=float, default=0.0003, metavar='G',
                    help='learning rate (default: 0.0003)')
parser.add_argument('--alpha', type=float, default=0.2, metavar='G',
                    help='Temperature parameter α determines the relative importance of the entropy\
                            term against the reward (default: 0.2)')
parser.add_argument('--automatic_entropy_tuning', type=bool, default=False, metavar='G',
                    help='Automaically adjust α (default: False)')
parser.add_argument('--seed', type=int, default=123456, metavar='N',
                    help='random seed (default: 123456)')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size (default: 256)')
parser.add_argument('--num_steps', type=int, default=1000001, metavar='N',
                    help='maximum number of steps (default: 1000000)')
parser.add_argument('--hidden_size', type=int, default=256, metavar='N',
                    help='hidden size (default: 256)')
parser.add_argument('--updates_per_step', type=int, default=1, metavar='N',
                    help='model updates per simulator step (default: 1)')
parser.add_argument('--start_steps', type=int, default=10000, metavar='N',
                    help='Steps sampling random actions (default: 10000)')
parser.add_argument('--target_update_interval', type=int, default=1, metavar='N',
                    help='Value target update per no. of updates per step (default: 1)')
parser.add_argument('--replay_size', type=int, default=1000000, metavar='N',
                    help='size of replay buffer (default: 10000000)')
parser.add_argument('--cuda', action="store_true",
                    help='run on CUDA (default: False)')
parser.add_argument('--param_beta', type=float, default=0.1, metavar='G',
                    help='Bonus reward regularizer')
parser.add_argument('--info', type=str, default='info_needed')
parser.add_argument('--cal_freq', type=int, default=500)
parser.add_argument('--HVD_cand', type=int, default=5)
parser.add_argument('--sampls', type=int, default=25)

args = parser.parse_args()

# Environment
# env = NormalizedActions(gym.make(args.env_name))
env = gym.make(args.env_name)
env.seed(args.seed)
env.action_space.seed(args.seed)

torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Agent
agent = SAC(env.observation_space.shape[0], env.action_space, args)

#Tensorboard
writer = SummaryWriter('runs_Hopper/{}_SAC_valfun_{}_{}_{}_{}_{}'.format(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"), args.env_name,
                                                             args.policy, args.info, args.seed,"autotune" if args.automatic_entropy_tuning else ""))

# Memory
memory = ReplayMemory(args.replay_size, args.seed)
enc_buffer = []

HVD_point = torch.zeros([1,2])# (x,y) y is the bottleneck for an env.

def trnf(sub_val):
    #return (2/(np.exp (sub_val)+  np.exp(- sub_val)))
    return (2/(math.exp (sub_val)+  math.exp(- sub_val)))

def rew_update(state, rew, tstep, bufr_enc, HVD_point, model, new_beta = args.param_beta, new_freq = args.cal_freq, samples = args.sampls ):
    state = state.reshape(1,-1)
    observation = torch.from_numpy(state.astype('float32'))

    enc_observation = model.encoder(observation) #encode observation
    candt = args.HVD_candt No. of candidate density points
    # set the no. of randomly sampled voters
    k = int(0.010 *len(bufr_enc)) # 10% data considered for neighborhood

    if tstep % new_freq == 0 and tstep >0:
        '''
        Calculate new HVD_point every 'new_freq' timestep
        '''
        print("\n Current HVD_point calculation frequency :"+str(new_freq))
        print("\n Current #samples from neighborhood: " +str(samples))
        print("\n Current beta :"+str(new_beta))
        HVD_point = updt.get_density_pk(candt,k,bufr_enc) # Get density peak
        print("HVD_point new :"+str(HVD_point))

    else:
        pass

    if HVD_point[0][0] == 0:
        rew_d = rew
    else:
        wts = nn.Softmax(dim=0)
        nos = samples
        bn = deque(state.shape)
        bn.appendleft(nos)
        samp_all = np.zeros((bn)) #'nos'rows
        #run a loop to create the samples by adding noise to state vector
        for ink in range(nos):
            samp_all[ink][:] = np.random.normal(0, 0.1, size=state.shape)+ state

        samp_all = np.concatenate((samp_all,state.reshape(1,state.shape[0],state.shape[1]))) # add original state to the array
        # find UCB2 values for all
        ucb_arr = np.zeros((nos+1, 1))

        for id,ovl in enumerate(samp_all):
            sum_V_score = [] #
            ovl = ovl.reshape(-1)
            #print(ovl.shape)
            #get V using original state (non encoded)
            saf = torch.from_numpy(ovl).float().to(device)
            val_V = agent.valfun(saf).cpu() #obtain V for given state vector

            o_enc =  torch.from_numpy(ovl.astype('float32'))

            o_enc = model.encoder(o_enc) #------------****
            nov_score =  torch.abs(torch.sum(torch.sub(HVD_point, o_enc.detach())))

            ucb_arr[id]= nov_score * val_V.detach()

        ucb_sub = np.subtract(np.max(ucb_arr,axis=0), ucb_arr[-1])#was -i ***
        rew_b = trnf(ucb_sub) # close gives 1, far gives less e.g.0.3

        rew_d = (1-new_beta)*rew + (new_beta * rew_b)
        rew_d = rew_d.item(0)
        rew_d =  torch.tensor(rew_d)

        #writer.add_scalar('V value', val_V, tstep)
    return rew_d, enc_observation, HVD_point

def evaluate_policy(total_numsteps):
    avg_reward = 0.
    episodes = 5
    for _  in range(episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)

            next_state, reward, done, _ = env.step(action)
            episode_reward += reward

            state = next_state
        avg_reward += episode_reward
    avg_reward /= episodes


    writer.add_scalar('avg_reward/test', avg_reward, total_numsteps)

    print("----------------------------------------")
    print("Test Episodes: {}, Avg. Reward: {}".format(episodes, round(avg_reward, 2)))
    print("----------------------------------------")


# Training Loop
total_numsteps = 0
updates = 0

for i_episode in itertools.count(1):
    episode_reward = 0
    episode_steps = 0
    done = False
    state = env.reset()
    model = updt.init_model()

    while not done:
        if args.start_steps > total_numsteps:
            action = env.action_space.sample()  # Sample random action
        else:
            action = agent.select_action(state)  # Sample action from policy

        # Evaluate current policy
        if total_numsteps % 5000 == 0 and args.eval is True:
            evaluate_policy(total_numsteps)

        if len(memory) > args.batch_size:
            # Number of updates per step in environment
            for i in range(args.updates_per_step):
                # Update parameters of all the networks
                critic_1_loss, critic_2_loss, policy_loss, ent_loss, alpha, val_fun_loss = agent.update_parameters(memory, args.batch_size, updates)
                #writer.add_scalar('V loss', val_fun_loss, updates)
                updates += 1

        next_state, reward, done, _ = env.step(action) # Step
        episode_steps += 1
        total_numsteps += 1
        episode_reward += reward

        # Ignore the "done" signal if it comes from hitting the time horizon.
        # (https://github.com/openai/spinningup/blob/master/spinup/algos/sac/sac.py)
        mask = 1 if episode_steps == env._max_episode_steps else float(not done)
        reward_updated, state_encoded, HVD_point = rew_update(next_state, reward, total_numsteps, enc_buffer, HVD_point, model)
        enc_buffer.append(state_encoded.detach().float()) #Save encoded state to buffer
        #print('\n Updated reward'+str(type(reward_updated)))
        memory.push(state, action, reward_updated, next_state, mask) # Append transition to memory, with updated reward

        state = next_state

    if total_numsteps > args.num_steps:
        break

    writer.add_scalar('reward/train', episode_reward, i_episode)
    print("Episode: {}, total numsteps: {}, episode steps: {}, reward: {}".format(i_episode, total_numsteps, episode_steps, round(episode_reward, 2)))


env.close()
