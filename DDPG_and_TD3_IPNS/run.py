import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import itertools
import gym
import argparse
import os
from torch.utils.tensorboard import SummaryWriter
import utils
#import TD3
import OurDDPG
import DDPG_2
import TD3_2
import get_rewupdate as updt
import math
import time
from random import random
from collections import namedtuple, deque

device = torch.device("cpu")
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default="TD3")                  # Policy name (TD3, DDPG or OurDDPG)
parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
parser.add_argument("--max_timesteps", default=1e6, type=int)   # Max time steps to run environment
parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
parser.add_argument("--discount", default=0.99)                 # Discount factor
parser.add_argument("--tau", default=0.005)                     # Target network update rate
parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
parser.add_argument("--save_model", action="store_true")        # Save model and optimizer parameters
parser.add_argument("--load_model", default="")                 # Model load file name, "" doesn't load, "default" uses file_name
parser.add_argument("--info", default="info_needed")
parser.add_argument('--param_beta', type=float, default=0.1, metavar='G', help='Bonus reward regularizer')
args = parser.parse_args()

file_name = f"{args.policy}_{args.env}_{args.seed}"
print("---------------------------------------")
print(f"Policy: {args.policy}, Env: {args.env}, Seed: {args.seed}")
print("---------------------------------------")

if not os.path.exists("./results"):
	os.makedirs("./results")

if args.save_model and not os.path.exists("./models"):
	os.makedirs("./models")

env = gym.make(args.env)
writer = SummaryWriter('runs_Hopper_TD3/TD3_2_{}_{}_{}'.format (args.env, args.info, args.seed))

enc_buffer = [] #Encoded state memory
HVD_point = torch.zeros([1,2])# (x,y) y is the bottleneck for an env.

# Set seeds
env.seed(args.seed)
env.action_space.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

state_dim = env.observation_space.shape[0]
action_dim = env.action_space.shape[0]
max_action = float(env.action_space.high[0])

kwargs = {
	"state_dim": state_dim,
	"action_dim": action_dim,
	"max_action": max_action,
	"discount": args.discount,
	"tau": args.tau,
}

#----------------------------------------------------------------------
def trnf(sub_val):
    #return (2/(np.exp (sub_val)+  np.exp(- sub_val)))
    return (2/(math.exp (sub_val)+  math.exp(- sub_val)))

def rew_update(state, rew, tstep, bufr_enc, HVD_point, model, new_beta = args.param_beta,new_freq = 500 ):
    state = state.reshape(1,-1)
    observation = torch.from_numpy(state.astype('float32'))

    enc_observation = model.encoder(observation) #encode observation ---***
    #enc_observation = observation

    candt = 5 # No. of candidate density peaks
    # set the no. of randomly sampled voters
    k = int(0.010 *len(bufr_enc)) # 10% data considered for neighborhood

    if tstep % new_freq == 0 and tstep >0:
        '''
        Calculate new HVD_point every 'new_freq' timestep
        '''
        print("\n Current HVD_point calculation frequency :"+str(new_freq))
        print("\n Current beta :"+str(new_beta))
        HVD_point = updt.get_HVD_point(candt,k,bufr_enc) # Get density peak
        print("HVD_point new :"+str(HVD_point))

    else:
        pass

    if HVD_point[0][0] == 0:
        rew_d = rew
    else:
        # Compare and Calculate state novelty score
        # generate samples around current state, gaussian with sd:0.1
        #wts = nn.Sigmoid()
        wts = nn.Softmax(dim=0)
        nos = 25 #number of samples
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
            val_V = param_updt.valfun(saf).cpu() #obtain V for given state vector#**
	    #val_V = param_updt.valfun(saf).device() #obtain V for given state vector

            o_enc =  torch.from_numpy(ovl.astype('float32'))
            o_enc = model.encoder(o_enc) #------------****
            nov_score =  torch.abs(torch.sum(torch.sub(HVD_point, o_enc.detach())))
            if tstep < 10000:
                ucb_arr[id]= nov_score
            else:
                ucb_arr[id]= nov_score * val_V.detach()

        ucb_sub = np.subtract(np.max(ucb_arr,axis=0), ucb_arr[-1])
        rew_b = trnf(ucb_sub) # close gives 1, far gives less e.g.0.3
        rew_d = (1-new_beta)*rew + (new_beta * rew_b)
        rew_d = rew_d.item(0)
        rew_d =  torch.tensor(rew_d)

    return rew_d, enc_observation, HVD_point

#------------------------------------------------------------------------

def eval_policy(policy, env_name, seed, total_numsteps, eval_episodes=5 ):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	avg_reward = 0.
	for _ in range(eval_episodes):
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			avg_reward += reward

	avg_reward /= eval_episodes

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {avg_reward:.3f}")
	print("---------------------------------------")
	writer.add_scalar('avg_reward/test',avg_reward, total_numsteps)

	return avg_reward


if __name__ == "__main__":

	# Initialize policy
	if args.policy == "TD3":
		# Target policy smoothing is scaled wrt the action scale
		kwargs["policy_noise"] = args.policy_noise * max_action
		kwargs["noise_clip"] = args.noise_clip * max_action
		kwargs["policy_freq"] = args.policy_freq
		policy = TD3_2.TD3(**kwargs)
		param_updt = TD3_2.Vnet(state_dim, action_dim, max_action)

	elif args.policy == "OurDDPG":
		policy = OurDDPG.DDPG(**kwargs)

	elif args.policy == "DDPG":
		policy = DDPG_2.DDPG(**kwargs)
		param_updt = DDPG_2.Vnet(**kwargs)

	if args.load_model != "":
		policy_file = file_name if args.load_model == "default" else args.load_model
		policy.load(f"./models/{policy_file}")

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)

	# Evaluate untrained policy
	#evaluations = [eval_policy(policy, args.env, args.seed)]

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	model = updt.init_model()

	for t in range(int(args.max_timesteps)):
		ra = random()

		episode_timesteps += 1

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action)

		#------------------reward update------------------------
		# Update reward using novelty bonus
		reward_updated, state_encoded, HVD_point = rew_update(next_state, reward, t, enc_buffer, HVD_point, model)
		enc_buffer.append(state_encoded.detach().float()) #Save encoded state to buffer
		#------------------------------------------------------------------

		#if ra < 0.3:
			#reward_updated = reward

		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0

		replay_buffer.add(state, action, next_state, reward_updated, done_bool)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train(replay_buffer, args.batch_size)
			# Update the value function irrespective of policy update
			val_loss = param_updt.update_parameters(state_dim, replay_buffer, args.batch_size)
			writer.add_scalar('val_fun training loss',val_loss, t)

		# Evaluate episode
		if t % 5000 == 0:
			eval_policy(policy, args.env, args.seed, t)

		#rest after sometime
		if t%200000 == 0 and t>0:
			time.sleep(600)
			print('\n Going to sleep...')


		if done:
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			#net1 = reward_updated.item()-reward
			#print('\n Net reward update:'+str(net1) )
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1
