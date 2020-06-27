import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np

from .network import BaseNetwork
from .replay_buffer import ReplayBuffer

class Agent:
    def __init__(self, args):
        actor_input_dim = args.obs_dim
        actor_out_dim = args.action_dim
        critic_input_dim = (args.obs_dim + args.action_dim) * args.num_agents
        critic_out_dim = 1
        self.actor = BaseNetwork(actor_input_dim, actor_out_dim, 
                                    hidden_dim=args.hidden_dim,
                                    normalize_input=args.normalize_input, 
                                    out_func=F.tanh)
        self.critic = BaseNetwork(critic_input_dim, critic_out_dim,
                                    hidden_dim=args.hidden_dim,
                                    normalize_input=args.normalize_input)
        self.target_actor = BaseNetwork(actor_input_dim, actor_out_dim, 
                                    hidden_dim=args.hidden_dim,
                                    normalize_input=args.normalize_input,
                                    out_func=F.tanh)
        self.target_critic = BaseNetwork(critic_input_dim, critic_out_dim,
                                    hidden_dim=args.hidden_dim,
                                    normalize_input=args.normalize_input)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optim = torch.optim.Adam(self.actor.parameters(), lr=args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic.parameters(), lr=args.lr_critic)  

        self.args = args


    def get_action(self, obs, is_target=False, decode=False):
        obs = torch.tensor(obs, dtype=torch.float32)
        # with torch.no_grad():
        if is_target:
            action = self.target_actor.forward(obs)
        else:
            action = self.actor.forward(obs)

        if decode:
            action = torch.argmax(action).numpy()
        else:
            softmax = torch.nn.Softmax(0)
            action = softmax(action)
        
        return action

    
    def get_q(self, x, is_target=False):
        x = torch.tensor(x, dtype=torch.float32)
        # with torch.no_grad():
        if is_target:
            q = self.target_critic.forward(x)
        else:
            q = self.critic.forward(x)
        return q


    def target_update(self, tau, actor=True):
        if actor:
            target = self.target_actor
            source = self.actor
        else:
            target = self.target_critic
            source = self.critic

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1 - tau) + param.data * tau)
    