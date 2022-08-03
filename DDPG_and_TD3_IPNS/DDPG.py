import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import datetime
import argparse

# Added a Value network, weights init and AE module

device = torch.device("cpu")

# Implementation of Deep Deterministic Policy Gradients (DDPG)
# Paper: https://arxiv.org/abs/1509.02971
# [Not the implementation used in the TD3 paper]

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super(Actor, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400, 300)
		self.l3 = nn.Linear(300, action_dim)

		self.max_action = max_action


	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super(Critic, self).__init__()

		self.l1 = nn.Linear(state_dim, 400)
		self.l2 = nn.Linear(400 + action_dim, 300)
		self.l3 = nn.Linear(300, 1)


	def forward(self, state, action):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(torch.cat([q, action], 1)))
		return self.l3(q)

#-----------------------------------------------------------------------------
'''
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)
'''


class ValueNetwork(nn.Module):
    def __init__(self, num_inputs):
        super(ValueNetwork, self).__init__()

        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 1)

        #self.apply(weights_init_)

    def forward(self, state):
        x = F.relu(self.linear1(state))
        x = F.relu(self.linear2(x))
        x = self.linear3(x)
        return x

class Vnet(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
		self.device = torch.device("cpu")
		self.valfun = ValueNetwork(state_dim).to(device=self.device)
		self.valfun_optim = torch.optim.Adam(self.valfun.parameters(), lr= 0.0003, weight_decay=1e-2)
		self.discount = discount
		self.tau = tau

	def update_parameters(self, state_dim, replay_buffer, batch_size):
		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# value function --------------------------------------------------------------
		V_next_state = self.valfun(next_state)
		valfun_target = reward + not_done *self.discount * V_next_state.squeeze(0)

		V = self.valfun(state)
		valfun_loss = F.mse_loss(valfun_target, V)
		self.valfun_optim.zero_grad()
		valfun_loss.backward()
		self.valfun_optim.step()

		return valfun_loss.item()
#------------------------------------------------------------------------------

class DDPG(object):
	def __init__(self, state_dim, action_dim, max_action, discount=0.99, tau=0.001):
		self.actor = Actor(state_dim, action_dim, max_action).to(device)
		self.actor_target = copy.deepcopy(self.actor)
		self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=1e-4)

		self.critic = Critic(state_dim, action_dim).to(device)
		self.critic_target = copy.deepcopy(self.critic)
		self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), weight_decay=1e-2)

		self.discount = discount
		self.tau = tau


	def select_action(self, state):
		state = torch.FloatTensor(state.reshape(1, -1)).to(device)
		return self.actor(state).cpu().data.numpy().flatten()


	def train(self, replay_buffer, batch_size=64):
		# Sample replay buffer
		state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

		# Compute the target Q value
		target_Q = self.critic_target(next_state, self.actor_target(next_state))
		target_Q = reward + (not_done * self.discount * target_Q).detach()

		# Get current Q estimate
		current_Q = self.critic(state, action)

		# Compute critic loss
		critic_loss = F.mse_loss(current_Q, target_Q)

		# Optimize the critic
		self.critic_optimizer.zero_grad()
		critic_loss.backward()
		self.critic_optimizer.step()

		# Compute actor loss
		actor_loss = -self.critic(state, self.actor(state)).mean()

		# Optimize the actor
		self.actor_optimizer.zero_grad()
		actor_loss.backward()
		self.actor_optimizer.step()

		# Update the frozen target models
		for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

		for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
			target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)


	def save(self, filename):
		torch.save(self.critic.state_dict(), filename + "_critic")
		torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")

		torch.save(self.actor.state_dict(), filename + "_actor")
		torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")


	def load(self, filename):
		self.critic.load_state_dict(torch.load(filename + "_critic"))
		self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
		self.critic_target = copy.deepcopy(self.critic)

		self.actor.load_state_dict(torch.load(filename + "_actor"))
		self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))
		self.actor_target = copy.deepcopy(self.actor)

#---------------------------------------------------------------------------------
# Define Autoencoder architecture
class AEnetwork(nn.Module):
    def __init__(self, num_inputs):
        super(AEnetwork, self).__init__()

        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(num_inputs, 64),
            torch.nn.ELU(),
            torch.nn.Linear(64,16),
            torch.nn.ELU(),
            torch.nn.Linear(16,5),
            torch.nn.ELU(),
            torch.nn.Sigmoid(),
        )

        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(5,16),
            torch.nn.ELU(),
            torch.nn.Linear(16,64),
            torch.nn.ELU(),
            torch.nn.Linear(64, num_inputs),
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
