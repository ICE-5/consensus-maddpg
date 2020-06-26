import torch
import os
import numpy as np

from network import BaseNetwork
from replay_buffer import ReplayBuffer
from common.utils import EpsilonNormalActionNoise

class Agent:
    def __init__(self, args):
        actor_input_dim = args.obs_dim
        actor_out_dim = args.action_dim
        critic_input_dim = (args.obs_dim + args.action_dim) * args.num_agents
        critic_out_dim = 1
        self.actor = BaseNetwork(actor_input_dim, actor_out_dim, 
                                    hidden_dim=args.hidden_dim,
                                    normalize_input=args.normalize_input,
                                    discrete_action=args.discrete_action)
        self.critic = BaseNetwork(critic_input_dim, critic_out_dim,
                                    hidden_dim=args.hidden_dim,
                                    normalize_input=args.normalize_input)
        self.target_actor = BaseNetwork(actor_input_dim, actor_out_dim, 
                                    hidden_dim=args.hidden_dim,
                                    normalize_input=args.normalize_input,
                                    discrete_action=args.discrete_action)
        self.target_critic = BaseNetwork(critic_input_dim, critic_out_dim,
                                    hidden_dim=args.hidden_dim,
                                    normalize_input=args.normalize_input)
        # TODO: discuss buffer config                          
        self.buffer = ReplayBuffer(args.buffer_size)
        self.args = args
        

    def select_action(self, obs):
        if np.random.uniform() < self.args.epsilon:
            action = torch.FloatTensor(self.args.action_dim).uniform_(self.args.action_bound_min, self.args.action_bound_max)
        else:
            obs = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            action = self.actor(obs).squeeze(0)
            noise = torch.FloatTensor(self.args.action_dim).normal_(0, self.args.norse_rate)
            action += noise
            action = torch.clamp(action, self.args.action_bound_min, self.args.action_bound_max)
            
        return action


