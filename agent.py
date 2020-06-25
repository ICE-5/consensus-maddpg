import numpy as np
import torch
import os
from maddpg.maddpg import MADDPG


class Agent:
    def __init__(self, agent_id, args):
        self.args = args
        self.agent_id = agent_id
        self.policy = MADDPG(args, agent_id)

    # def select_action(self, o, noise_rate, epsilon):
    #     import ipdb;
    #     ipdb.set_trace()
    #     if np.random.uniform() < epsilon:
    #         u = np.random.uniform(-self.args.high_action, self.args.high_action, self.args.action_shape[self.agent_id])
    #     else:
    #         inputs = torch.tensor(o, dtype=torch.float32, device = self.args.device).unsqueeze(0)
    #         pi = self.policy.actor_network(inputs).squeeze(0)
    #         # print('{} : {}'.format(self.name, pi))
    #         u = pi.cpu().numpy()
    #         noise = noise_rate * self.args.high_action * np.random.randn(*u.shape)  # gaussian noise
    #         u += noise
    #         u = np.clip(u, -self.args.high_action, self.args.high_action)
    #     return u.copy()

    def select_action(self, o, noise_rate, epsilon):
        if torch.rand(1) < epsilon:
            u = torch.cuda.FloatTensor(self.args.action_shape[self.agent_id]).uniform_(-self.args.high_action,
                                                                                       self.args.high_action)
        else:
            inputs = torch.tensor(o, dtype=torch.float32, device = self.args.device).unsqueeze(0)
            u = self.policy.actor_network(inputs).squeeze(0)
            noise = torch.tensor(noise_rate * self.args.high_action * torch.randn(u.size()[0]), device = self.args.device)# gaussian noise
            u += noise
            u = torch.clamp(u, -self.args.high_action, self.args.high_action)
        return u

    def learn(self, transitions, other_agents):
        self.policy.train(transitions, other_agents)

