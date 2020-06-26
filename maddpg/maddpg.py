import torch
import os

from agent import Agent

class MADDPG:
    def __init__(self, args, env):
        self.num_agent = args.num_agent
        self.agents = [Agent(args) for i in range(self.num_agent)]
        self.env = env
        self.args = args
    
    def get_transitions(self):
        while not done:
            # TODO: translate pseudocode
            ## reset env
            ## get initial collective state from env, state_0
            curr_obs = state_0
            actions = torch.empty(self.args.action_dim * self.args.num_agents)
            for i, agent in enumerate(self.agents):
                ## slice obs of this agent
                ## obs = curr_obs[i]
                action = agent.select_action(obs)
                ## combine into collective action
                ## get next obs by interacting with env
                
            ## store transition


