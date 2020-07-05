import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .network import Actor, Critic
from .replay_buffer import ReplayBuffer

class Consensus:
    def __init__(self, args):
        self.device = args.device
        self.K = args.common_agents
        obs_dim = args.obs_dim_arr[0]
        act_dim = args.act_dim
        obs_dim_n = obs_dim * self.K
        act_dim_n = act_dim * self.K

        self.critic = Critic(obs_dim_n, act_dim_n, hidden_dim=args.hidden_dim)
        self.target_critic = Critic(obs_dim_n, act_dim_n, hidden_dim=args.hidden_dim)

        self.critic.to(args.device)
        self.target_critic.to(args.device)

        self.target_critic.load_state_dict(self.critic.state_dict())

        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)

        self.args = args

    def get_q(self, obs, act, is_target=False):
        obs = torch.FloatTensor(obs).to(self.device)
        act = torch.FloatTensor(act).to(self.device)
        if is_target:
            q = self.target_critic.forward(obs, act)
        else:
            q = self.critic.forward(obs, act)
        return q

    def target_update(self, tau):
        target = self.target_critic
        source = self.critic
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)