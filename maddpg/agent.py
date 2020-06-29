import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .network import Actor, Critic
from .replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, args, agent_id):
        self.device = args.device

        obs_dim = args.obs_dim_arr[agent_id]
        act_dim = args.act_dim
        obs_dim_n = np.sum(args.obs_dim_arr)
        act_dim_n = act_dim * args.num_agents

        self.actor = Actor(obs_dim, act_dim, hidden_dim=args.hidden_dim)
        self.critic = Critic(obs_dim_n, act_dim_n, hidden_dim=args.hidden_dim)
        self.target_actor = Actor(obs_dim, act_dim, hidden_dim=args.hidden_dim)
        self.target_critic = Critic(obs_dim_n, act_dim_n, hidden_dim=args.hidden_dim)

        self.actor.to(args.device)
        self.critic.to(args.device)
        self.target_actor.to(args.device)
        self.target_critic.to(args.device)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        self.buffer = ReplayBuffer(args, obs_dim, act_dim)

        # TODO: check necessity
        self.args = args


    def get_action(self, obs, is_target=False, is_argmax=False):
        # import ipdb
        # ipdb.set_trace()
        # obs = torch.FloatTensor(obs)
        obs = torch.FloatTensor(obs).to(self.device)

        # with torch.no_grad():
        if is_target:
            act = self.target_actor.forward(obs)
        else:
            act = self.actor.forward(obs)
        softmax = torch.nn.Softmax(0)
        act = softmax(act)

        if is_argmax:
            act = torch.argmax(act)
        return act

    
    def get_q(self, obs, act, is_target=False):
        # obs = torch.FloatTensor(obs)
        # act = torch.FloatTensor(act)
        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)

        # with torch.no_grad():
        if is_target:
            q = self.target_critic.forward(obs, act)
        else:
            q = self.critic.forward(obs, act)
        return q


    def target_update(self, tau, is_actor=True):
        if is_actor:
            target = self.target_actor
            source = self.actor
        else:
            target = self.target_critic
            source = self.critic

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
    
