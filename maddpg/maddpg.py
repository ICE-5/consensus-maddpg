import torch
import torch.nn as nn
import torch.nn.functional as F
import os, time
import numpy as np
from tqdm import tqdm

from .agent import Agent
from .replay_buffer import ReplayBuffer
from utils.helper import *
from copy import deepcopy

class MADDPG:
    def __init__(self, args, env):
        args.device = torch.device('cuda' if torch.cuda.is_available() and args.device == "gpu" else 'cpu')
        self.args = args
        self.n = args.num_agents
        self.act_dim = args.act_dim
        self.obs_dim_arr = args.obs_dim_arr

        self.batch_size = args.batch_size
        self.num_episodes = args.num_episodes
        self.max_episode_len = args.max_episode_len

        self.burnin_size = args.burnin_size

        self.update_rate = args.update_rate
        self.target_update_rate = args.target_update_rate
        
        self.gamma = args.gamma
        self.tau = args.tau

        self.env = env
        self.agents = [ Agent(args, i) for i in range(self.n) ]

        self.train_critic_loss = []
        self.train_actor_loss = []

        self.eval_rate = args.evaluate_num_episodes

    
    def memory_burnin(self):
        start = time.time()
        print(f'-----> replay buffer. start burn-in memory.')
        counter = 0
        while counter < self.burnin_size:
            done = False
            curr_obs_n = self.env.reset()
            while not done:
                act_n = [self.env.action_space[i].sample() for i in range(len(self.env.action_space))]
                act_n = one_hot(act_n, self.act_dim)
                next_obs_n, reward_n, done_n, _ = self.env.step(act_n)

                for i, agent in enumerate(self.agents):
                    agent.buffer.add(curr_obs_n[i], act_n[i], next_obs_n[i], reward_n[i], done_n[i])

                curr_obs_n = deepcopy(next_obs_n)
                
                counter +=1
                done = all(done_n)
                if done or counter >= self.burnin_size:
                    break
        print(f'-----> replay buffer. finish burn-in memory.\t\tburn-in size: {counter}\ttime: {time.time()-start}')


    def get_minibatch(self):
        idxs = self.agents[0].buffer.sample_minibatch(self.batch_size)
        for idx, agent in enumerate(self.agents):
            component = agent.buffer.get_minibatch_component(idxs)
            if idx==0:
                minibatch = component
            else:
                minibatch = [ torch.cat((item, component[i]), dim=1) for i, item in enumerate(minibatch) ]
        return minibatch
            

    def agent_critic_loss(self, agent_id, minibatch):
        curr_obs_n, act_n, next_obs_n, reward_n, _ = minibatch
        o = next_obs_n
        a = torch.empty((self.batch_size, self.n * self.act_dim))
        reward_n = reward_n.to(self.args.device)
        
        for i, agent in enumerate(self.agents):
            curr_obs = self._local_slice(i, curr_obs_n)
            act = agent.get_action(curr_obs, is_target=True, is_argmax=False)
            a = self._local_replace(i, act, a, is_action=True)

        y = self.agents[agent_id].get_q(o, a, is_target=True)
        y = self.gamma * y + reward_n[:, agent_id]

        # calculate goal_y
        goal_o = curr_obs_n
        goal_a = act_n
        goal_y = self.agents[agent_id].get_q(goal_o, goal_a, is_target=False)
        
        return F.mse_loss(goal_y, y)
    

    def agent_actor_loss(self, agent_id, minibatch):
        curr_obs_n, act_n, _, _, _ = minibatch
        ### o
        o = curr_obs_n
        ### a
        a = act_n
        agent_curr_obs = self._local_slice(agent_id, curr_obs_n)
        agent_act = self.agents[agent_id].get_action(agent_curr_obs, is_target=False, is_argmax=False)
        a = self._local_replace(agent_id, agent_act, a, is_action=True)

        loss = - self.agents[agent_id].get_q(o, a, is_target=False).mean()
        return loss

    
    def agent_update(self, agent_id, step):
        if not step % self.update_rate == 0:
            return 0., 0.
        # get minibatch sample
        minibatch = self.get_minibatch()
        # critic
        agent_critic_loss = self.agent_critic_loss(agent_id, minibatch)
        self.agents[agent_id].critic_optim.zero_grad()
        agent_critic_loss.backward()
        self.agents[agent_id].critic_optim.step()
        # actor
        agent_actor_loss = self.agent_actor_loss(agent_id, minibatch)
        self.agents[agent_id].actor_optim.zero_grad()
        agent_actor_loss.backward()
        self.agents[agent_id].actor_optim.step()
        return agent_critic_loss.item(), agent_actor_loss.item()
    

    def agent_target_update(self, agent_id, tau, step):
        if not step % self.target_update_rate == 0:
            return
        self.agents[agent_id].target_update(tau, is_actor=True)
        self.agents[agent_id].target_update(tau, is_actor=False)

    def train(self):
        step = 0
        total_critic_loss, total_actor_loss = 0., 0.
        rewards, rewards_frd, rewards_adv = [], [], []
        for episode in tqdm(range(self.num_episodes)):
            curr_obs_n = self.env.reset()
            episode_len = 0
            while True:

                act_n = [agent.get_action(curr_obs_n[i], is_target=False, is_argmax=True) for i, agent in enumerate(self.agents)]     
                act_n = one_hot(act_n, self.act_dim)
                next_obs_n, reward_n, done_n, _ = self.env.step(act_n)

                # TODO: add noise
                act_n = add_noise(act_n)

                for i in range(self.n):
                    self.agents[i].buffer.add(curr_obs_n[i], act_n[i], next_obs_n[i], reward_n[i], done_n[i])
                curr_obs_n = deepcopy(next_obs_n)

                step += 1
                episode_len += 1
                done = all(done_n)
                terminal = (episode_len >= self.max_episode_len)

                for i in range(self.n):
                    critic_loss, actor_loss = self.agent_update(i, step)
                    total_critic_loss += critic_loss
                    total_actor_loss  += actor_loss
                
                for i in range(self.n):
                    self.agent_target_update(i, self.tau, step)

                if step % self.update_rate == 0:
                    # print(f'episode: {episode}\t step: {step}\t\t critic loss: {total_critic_loss}\t actor loss: {total_actor_loss}')
                    total_critic_loss, total_actor_loss = 0., 0.
                
                if done or terminal:
                    break

            if episode % self.args.evaluate_rate == 0:
                r_avg, r_frd, r_adv = self.evaluate()
                rewards.append(r_avg)
                rewards_frd.append(r_frd)
                rewards_adv.append(r_adv)
                print(f"episode: {episode}\t total reward :{r_avg}\t friend reward :{r_frd}\t adv reward :{r_adv}")

    '''
    Return three rewards: average reward among episodes, average reward among episodes and num_friends, average reward among episodes and num_advesaries, 
    '''
    def evaluate(self):
        rewards_frd, rewards_adv, rewards = [], [], []
        for episodes in range(self.args.evaluate_num_episodes):
            curr_obs_n = self.env.reset()
            episode_len = 0
            reward_frd, reward_adv, reward = 0., 0., 0.
            while True:
                if self.args.render:
                    self.env.render()
                act_n = [self.agents[i].get_action(curr_obs_n[i], is_target=False, is_argmax=True) for i in range(self.n)]
                act_n = one_hot(act_n, self.act_dim)
                curr_obs_n, reward_n, done_n, _ = self.env.step(act_n)
                # Assume first k are friends, then the rest are adversaries.
                if self.args.num_adversaries != 0:
                    reward_adv += sum(reward_n[self.args.num_friends:])
                reward_frd += sum(reward_n[: self.args.num_friends + 1])
                reward = reward_frd + reward_adv

                episode_len += 1
                done = all(done_n)
                terminal = episode_len > self.args.evaluate_episode_len
                if done or terminal:
                    rewards.append(reward)
                    rewards_frd.append(reward_frd)
                    rewards_adv.append(reward_adv)
                    break
                # print(f"evaluate {episodes}, len {episode_len}")

        return sum(rewards) / self.args.evaluate_num_episodes, \
               sum(rewards_frd) / (self.args.evaluate_num_episodes * self.args.num_friends), \
               sum(rewards_adv) / (self.args.evaluate_num_episodes * self.args.num_adversaries)


    def _location(self, idx, is_action=False):
        if is_action:
            s = self.act_dim * idx
            e = self.act_dim * (idx+1)
        else:
            s = np.sum(self.obs_dim_arr[ : idx])
            e = np.sum(self.obs_dim_arr[ : (idx+1)])
        return int(s), int(e)


    def _local_slice(self, idx, target, is_action=False, is_batch=True):
        s, e = self._location(idx, is_action=is_action)
        if is_batch:
            return target[:, s:e]
        else:
            return target[s:e]
        

    def _local_replace(self, idx, source, target, is_action=False, is_batch=True):
        s, e = self._location(idx, is_action=is_action)
        if is_batch:
            target[:, s:e] = source
        else:
            target[s:e] = source
        return target


            
    
                
                



