import numpy as np
import random
import argparse
import math
import os
import itertools
import time
import gym
from collections import namedtuple, deque

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal, MultivariateNormal
from torch.utils.tensorboard import SummaryWriter

import helper as hp
import utility as ut

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=32, init_w=3e-3, log_std_min=-20, log_std_max=2):
        super(Actor, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)

        self.mu = nn.Linear(hidden_size, action_size)
        self.log_std_linear = nn.Linear(hidden_size, action_size)


    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.mu.weight.data.uniform_(-init_w, init_w)
        self.log_std_linear.weight.data.uniform_(-init_w, init_w)

    def forward(self, state):

        x = F.relu(self.fc1(state), inplace=True)
        x = F.relu(self.fc2(x), inplace=True)
        mu = self.mu(x)

        log_std = self.log_std_linear(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mu, log_std

    def evaluate(self, state, epsilon=1e-6):
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e = dist.sample().to(device)
        action = torch.tanh(mu + e * std)
        log_prob = Normal(mu, std).log_prob(mu + e * std) - torch.log(1 - action.pow(2) + epsilon)

        return action, log_prob


    def get_action(self, state):
        """
        returns the action based on a squashed gaussian policy, i.e. samples are obtained according to:
        a(s,e)= tanh(mu(s)+sigma(s)+e)
        """
        mu, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(0, 1)
        e      = dist.sample().to(device)
        action = torch.tanh(mu + e * std).cpu()
        return action[0]

    def get_act_detr(self, state): #Get deterministic action
        """
        Returns the mean action of the Gaussian Policy, without considering the std
        """
        mu, log_std = self.forward(state)
        action = torch.tanh(mu).cpu()
        return action[0]

class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, seed, hidden_size=32):
        super(Critic, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size+action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state, action):
        """Critic network"""
        x = torch.cat((state, action), dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent():

    def __init__(self, state_size, action_size, random_seed, hidden_size, action_prior="uniform"):

        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(random_seed)

        self.target_entropy = -action_size  # -dim(A)
        self.alpha = 1
        self.log_alpha = torch.tensor([0.0], requires_grad=True)
        self.alpha_optimizer = optim.Adam(params=[self.log_alpha], lr=LR_ACTOR)
        self._action_prior = action_prior

        print("Using: ", device)

        # Actor Network
        self.actor_local = Actor(state_size, action_size, random_seed, hidden_size).to(device)
        self.actor_optimizer = optim.Adam(self.actor_local.parameters(), lr=LR_ACTOR)

        # Critic Network (w/ Target Network)
        self.critic1 = Critic(state_size, action_size, random_seed, hidden_size).to(device)
        self.critic2 = Critic(state_size, action_size, random_seed, hidden_size).to(device)

        self.critic1_target = Critic(state_size, action_size, random_seed,hidden_size).to(device)
        self.critic1_target.load_state_dict(self.critic1.state_dict())

        self.critic2_target = Critic(state_size, action_size, random_seed,hidden_size).to(device)
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=LR_CRITIC, weight_decay=0)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=LR_CRITIC, weight_decay=0)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, random_seed) #Initialize Replay Buffer


    def step(self, state, action, reward, next_state, done, step, curew, intmt_updt_flag = 0):
        """Save experience in replay memory, and use random sample from buffer to learn."""

        if intmt_updt_flag == 0:
            # Just add data to primary replay buffer & run network update
            self.memory.add(state, action, reward, next_state, done, curew)
        else:
            pass

        dat =  [state, action, reward, next_state, done] #Get the current transition data
        if len(self.memory) > 2*BATCH_SIZE: # If buffer size enough for sampling
            experiences = self.memory.sample()

            if intmt_updt_flag == 2:
                for _ in range (len(experiences)):
                    experiences[_][0] = torch.from_numpy(np.array(dat[_]))
                self.learn(step, experiences, GAMMA)

            elif intmt_updt_flag == 1 or intmt_updt_flag == 0 :
                self.learn(step, experiences, GAMMA)
        else:
            pass


    def act(self, state):
        """Returns actions for given state as per current policy."""
        state = torch.from_numpy(state).float().to(device)
        action = self.actor_local.get_action(state).detach()
        return action

    def detr_act(self, state): # Custom
        """Returns deterministic actions for given state as per mean current policy."""
        state = torch.from_numpy(state).float().to(device)
        action = self.actor_local.get_act_detr(state).detach()
        return action

    def learn(self, step, experiences, gamma, d=1):
        """Updates actor, critics and entropy_alpha parameters using given batch of experience tuples.
        Parameters
        ======
            experiences (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences


        # ---------------------------- update critic ---------------------------- #
        # Get predicted next-state actions and Q values from target models
        next_action, log_pis_next = self.actor_local.evaluate(next_states)

        Q_target1_next = self.critic1_target(next_states.to(device), next_action.squeeze(0).to(device))
        Q_target2_next = self.critic2_target(next_states.to(device), next_action.squeeze(0).to(device))

        # take the mean of both critics for updating
        Q_target_next = torch.min(Q_target1_next, Q_target2_next)

        if FIXED_ALPHA == None:
            # Compute Q targets for current states (y_i)
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - self.alpha * log_pis_next.squeeze(0).cpu()))
        else:
            Q_targets = rewards.cpu() + (gamma * (1 - dones.cpu()) * (Q_target_next.cpu() - FIXED_ALPHA * log_pis_next.squeeze(0).cpu()))

        # Compute critic loss
        Q_1 = self.critic1(states, actions).cpu()
        Q_2 = self.critic2(states, actions).cpu()
        critic1_loss = 0.5*F.mse_loss(Q_1, Q_targets.detach())
        critic2_loss = 0.5*F.mse_loss(Q_2, Q_targets.detach())

        # Update critics
        # critic 1
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        # critic 2
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        if step % d == 0:
        # ---------------------------- update actor ---------------------------- #
            if FIXED_ALPHA == None:
                alpha = torch.exp(self.log_alpha)
                # Compute alpha loss
                actions_pred, log_pis = self.actor_local.evaluate(states)
                alpha_loss = - (self.log_alpha.cpu() * (log_pis.cpu() + self.target_entropy).detach().cpu()).mean()
                self.alpha_optimizer.zero_grad()
                alpha_loss.backward()
                self.alpha_optimizer.step()

                self.alpha = alpha
                # Compute actor loss
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0

                actor_loss = (alpha * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu() - policy_prior_log_probs ).mean()
            else:

                actions_pred, log_pis = self.actor_local.evaluate(states)
                if self._action_prior == "normal":
                    policy_prior = MultivariateNormal(loc=torch.zeros(self.action_size), scale_tril=torch.ones(self.action_size).unsqueeze(0))
                    policy_prior_log_probs = policy_prior.log_prob(actions_pred)
                elif self._action_prior == "uniform":
                    policy_prior_log_probs = 0.0

                actor_loss = (FIXED_ALPHA * log_pis.squeeze(0).cpu() - self.critic1(states, actions_pred.squeeze(0)).cpu()- policy_prior_log_probs ).mean()
            # Minimize the loss
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # ----------------------- update target networks ----------------------- #
            self.soft_update(self.critic1, self.critic1_target, TAU)
            self.soft_update(self.critic2, self.critic2_target, TAU)



    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model: PyTorch model (weights will be copied from)
            target_model: PyTorch model (weights will be copied to)
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)

class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "curew"])
        #self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, curew):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, curew)
        self.memory.append(e)

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        #experiences = random.sample(self.memory, k=self.batch_size)
        global meas
        global last_meas
        global avg_meas
        global th_val

        p = []
        q = []

        # sample double the batch size
        samp_size = 2* self.batch_size
        more_exp = random.sample(self.memory, k= samp_size)

        for m in range (len(more_exp)):
            if m < self.batch_size:
                p.append(more_exp[m].curew)
                #print("\n"+str(p))
            else:
                q.append(more_exp[m].curew)

        # Insert similarity constraint here
        f = hp.similarity() # instantiate
        meas = f.cosine_similarity(p,q) # call the similarity measure function
        last_meas.append(meas)
        avg_meas = np.mean(last_meas)

        if meas < 0.5: #SDP threshold
            # Sort the exp samples from max to min score
            sorted_more_exp = sorted(more_exp, key = lambda i: i.curew, reverse = True)
            experiences = sorted_more_exp[:self.batch_size] #Choose from Sorted samples

        elif meas >= 0.5:#SDP threshold
            experiences = more_exp[:self.batch_size] # Choose from unsorted samples


        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

'''
Run mean policy for % evaluation episodes and return/ save the  average episodic return
'''

def detr_eval():
    """Evaluate the current dterministic policy"""
    score_eval =0
    state = env.reset()
    epi_scores = []
    #state = state.reshape((1,state_size))
    for e in range (5): # Run 5 repeats of evaluation episode
        state = env.reset()
        state = state.reshape((1,state_size))
        score1 = []
        for t1 in range(1000):
            action = agent.detr_act(state)
            action_v = action
            action_v = np.clip(action_v*action_high, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            next_state = next_state.reshape((1,state_size))
            state = next_state
            #print('Eval reward:'+str(reward))
            score1.append(reward) # Score is cumulative reward
            if done:
                score_eval = np.sum(score1)
                break
        epi_scores.append(score_eval)
    avg_epi_score.append(np.mean(epi_scores))
    print("\n Epi rewards this eval run:"+str(epi_scores))
    print("\n Eval average score "+str(avg_epi_score))
    writer.add_scalar("Eval_episodic reward", np.mean(epi_scores), tsteps)


#------------------------Modified SAC-------------------------------------------------------------------
def filter_epi(epi_rew_list, epi_wise_data, prdct):
    ch =0
    global count
    global Eps2
    global cthres_step
    rn2 = np.random.random() # generate a random number

    if rn2 < Eps2:
        pass
    elif rn2 >= Eps2:
        # Pass all data to buffer
        for inds in range (len(epi_rew_list)):
            dicta = epi_wise_data[inds]
            # Should have a list of cumulative rewards
            curew = epi_rew_list[inds]
            ch+=1
            for k in range(len(dicta)): # save this episode's data to buffer
                agent.step(dicta[k]['state'],dicta[k]['action'], dicta[k]['reward'],dicta[k]['next_state'],dicta[k]['done'],dicta[k]['step'], curew)

def SAC(l,epi_count, max_t=1000, print_every=10):
    window = 20
    global pred
    global tsteps
    cmp = 0.0002
    sam = namedtuple("round_episaved", field_names=["state", "action", "reward", "next_state", "done","step"])
    all_epi_data = []
    epi_rew_list = []

    for i_episode in range(epi_count+1,epi_count+11): #was 11

        state = env.reset()

        state = state.reshape((1,state_size))
        score = 0

        one_epi_data = []
        rew = 0

        for t in range(max_t):

            # Run Evaluation episodes every nth timestep
            if tsteps%1000 == 0:
                detr_eval()

            action = agent.act(state)
            action_v = action
            action_v = np.clip(action_v*action_high, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            next_state = next_state.reshape((1,state_size))
            pv = np.random.random()

            if l < 2: # Do this for initial rounds
                agent.step(state, action, reward, next_state, done, t,-100,intmt_updt_flag = 0) # Both update and save to primary buffer

            elif l >= 2 and pv < Eps:
                agent.step(state, action, reward, next_state, done, t, 0, intmt_updt_flag = 2) # concat with regular updates, do not save

            elif l >= 2 and pv > Eps:
                agent.step(state, action, reward, next_state, done, t, 0, intmt_updt_flag = 1) # regular updates-no on-policy data, do not save

            else:
                pass

            v = sam(state, action, reward, next_state, done, t)._asdict() # each state transition as a dictionary tuple
            one_epi_data.append(v) # List of dictionaries of state transition data
            state = next_state
            score += reward # Score is cumulative reward
            tsteps = tsteps +1

            if done:
                epi_rew_list.append(score) # collect the episodic rewards over last 10 episodes
                tmill = tsteps/1000000
                #display
                print('\n Current tsteps in Million:' +str(tmill))
                print('\n Current tsteps:' +str(tsteps))
                break

        scores_deque.append(score)
        writer.add_scalar("average_tstep", np.mean(scores_deque), tsteps) # recording episodic rewards against tsteps
        average_100_scores.append(np.mean(scores_deque))

        print('\rEpisode {} Reward: {:.2f}  Average_last_100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)), end="")
        if i_episode % print_every == 0:
            print('\rEpisode {}  Reward: {:.2f}  Average_last_100 Score: {:.2f}'.format(i_episode, score, np.mean(scores_deque)))

        #--------------------------------------------------------------------------------------------------------------------------
        curew = score # This episode's cumulative reward

        # Keep track of cumulative rewards, for testing purposes (SMA value)
        if i_episode < window:
            arr.append(curew) #Add episodic cumulative rewards  for the first time
        else:
            pred = np.mean(arr)
            if curew > pred or math.isclose(curew, pred, abs_tol = 50): #abs_tot is env. specific
                arr.popleft()
                arr.append(curew)
            else:
                arr.popleft()
                diff = np.absolute(pred-curew)
                repl_val= pred*np.exp(-cmp*diff)
                arr.append(repl_val)
        #------------------------------------------------------------------------------------------

        all_epi_data.append(one_epi_data) # collect the episodic transition data over last 10 episodes

    epi_wise_data =list(itertools.chain(all_epi_data)) #convert list of list to a episodic list of items

    print("\n SMA at beginnig of next round: "+str(pred))
    
    if l >= 2 :
        filter_epi(epi_rew_list, epi_wise_data, pred) # one round worth data sent to filter method for selection, save to buffer and update
    else:
        pass

    svpath = '{}_seed_{}.pt'.format(args.info, args.seed)
    torch.save(agent.actor_local.state_dict(), svpath)

#------------------------**End Modified SAC---------------------------------------------------------------


def play():
    agent.actor_local.eval()
    for i_episode in range(1):

        state = env.reset()
        state = state.reshape((1,state_size))

        while True:
            action = agent.act(state)
            action_v = action[0].numpy()
            action_v = np.clip(action_v*action_high, action_low, action_high)
            next_state, reward, done, info = env.step(action_v)
            next_state = next_state.reshape((1,state_size))
            state = next_state

            if done:
                break




parser = argparse.ArgumentParser(description="")
parser.add_argument("-env", type=str,default="Pendulum-v0", help="Environment name")
parser.add_argument("-info", type=str, help="Information or name of the run")
parser.add_argument("-ep", type=int, default=10, help="The amount of training rounds, 1 round = 10 episodes")
parser.add_argument("-seed", type=int, default=0, help="Seed for the env and torch network weights, default is 0")
parser.add_argument("-lr", type=float, default=5e-4, help="Learning rate of adapting the network weights, default is 5e-4")
parser.add_argument("-a", "--alpha", type=float, help="entropy alpha value, if not choosen the value is leaned by the agent")
parser.add_argument("-layer_size", type=int, default=256, help="Number of nodes per neural network layer, default is 256")
parser.add_argument("-repm", "--replay_memory", type=int, default=int(1e6), help="Size of the Replay memory, default is 1e6")
parser.add_argument("--print_every", type=int, default=100, help="Prints every x episodes the average reward over x episodes")
parser.add_argument("-bs", "--batch_size", type=int, default=100, help="Batch size, default is 256")
parser.add_argument("-t", "--tau", type=float, default=1e-2, help="Softupdate factor tau, default is 1e-2")
parser.add_argument("-g", "--gamma", type=float, default=0.99, help="discount factor gamma, default is 0.99")
parser.add_argument("--saved_model", type=str, default=None, help="Load a saved model to perform a test run!")
args = parser.parse_args()

meas = []

if __name__ == "__main__":
    env_name = args.env
    seed = args.seed

    n_episodes = args.ep # number of episodes to be run

    round_len = 10 # episodes per round # was 10

    n_rounds = int((args.ep)/round_len) # number of rounds
    window = 20
    Eps = 1
    tsteps = 0

    GAMMA = args.gamma
    TAU = args.tau
    HIDDEN_SIZE = args.layer_size
    BUFFER_SIZE = int(args.replay_memory)
    BATCH_SIZE = args.batch_size        # minibatch size
    LR_ACTOR = args.lr         # learning rate of the actor
    LR_CRITIC = args.lr        # learning rate of the critic
    FIXED_ALPHA = args.alpha
    saved_model = args.saved_model
    bsf_curew = 0
    pred = None
    arr = deque(maxlen= window)
    scores_deque = deque(maxlen=100)
    last_meas = deque(maxlen=5)
    avg_meas = None
    average_100_scores = []
    avg_epi_score = []

    tar_tsteps = 100000 #--- was 100000 for IPD
    #bthres_step = 0.3
    Eps2 = 0 #was 0.3
    cthres_step = 0.7
    mthres_step = 0.001
    th_val = 0.3


    t0 = time.time()
    writer = SummaryWriter("runs/{}_seed_{}".format(args.info, args.seed))
    env = gym.make(env_name) #Initialize the Gym environmnet


    action_high = env.action_space.high[0]
    action_low = env.action_space.low[0]
    torch.manual_seed(seed)
    env.seed(seed)
    np.random.seed(seed)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]
    agent = Agent(state_size=state_size, action_size=action_size, random_seed=seed,hidden_size=HIDDEN_SIZE, action_prior="uniform") #"normal"

    if saved_model != None:
        agent.actor_local.load_state_dict(torch.load(saved_model)) # Save the learned model
        play()
    else:
        # Run for n rounds, where each round = 10 episodes
        for l in range (n_rounds):
            if tsteps <= tar_tsteps : # Million timesteps
                print("\n Round : "+str(l))
                print("\n Current tstep: "+str(tsteps))
                epi_count = l*round_len # e.g. 0..1*10..2*10
                SAC(l,epi_count, max_t=1000, print_every=args.print_every) # Run SAC algorithm, modified

    t1 = time.time()
    env.close()
    print("\n End of 1M timesteps. Training took {} min!".format((t1-t0)/60))
