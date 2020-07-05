import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
import os, time
import numpy as np
from tqdm import tqdm

from .agent import Agent
from .consensus import Consensus
from .replay_buffer import ReplayBuffer
from utils.helper import *
from copy import deepcopy

class MADDPG:
    def __init__(self, args, env, writer=None):
        args.device = torch.device('cuda' if torch.cuda.is_available() and args.device == "gpu" else 'cpu')
        self.args = args
        # Parameters for environment
        self.env = env
        self.n = args.num_agents
        self.act_dim = args.act_dim
        self.obs_dim_arr = args.obs_dim_arr
        self.noise_rate = args.noise_rate
        # Parameters for MADDPG network
        self.agents = [Agent(args, i) for i in range(self.n)]
        self.K = args.common_agents
        self.batch_size = args.batch_size
        self.num_episodes = args.num_episodes
        self.max_episode_len = args.max_episode_len
        self.burnin_size = args.burnin_size
        self.update_rate_maddpg = args.update_rate_maddpg
        self.target_update_rate_maddpg = args.target_update_rate_maddpg
        self.update_rate_consensus = args.update_rate_consensus
        self.target_update_rate_consensus = args.target_update_rate_consensus
        self.gamma = args.gamma
        self.tau = args.tau
        # Parameters for consensus networks
        self.cons = [Consensus(args) for _ in range(args.num_team)]
        # Parameters for evaluation
        self.writer = writer
        self.train_critic_loss = []
        self.train_actor_loss = []
        self.eval_rate = args.evaluate_num_episodes
        # Parameters for saveing/loading models
        self.path = args.save_dir
        # if not os.path.exists(self.path):
        #     os.makedirs(self.path)
        # self.load_models()

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

    def train(self):
        step = 0
        rewards, rewards_frd, rewards_adv = [], [], []

        for episode in tqdm(range(self.num_episodes)):
            curr_obs_n = self.env.reset()
            episode_len = 0
            while True:
                act_n = [agent.get_action(curr_obs_n[i], is_target=False, is_argmax=True) for i, agent in enumerate(self.agents)]
                act_n = one_hot(act_n, self.act_dim)
                next_obs_n, reward_n, done_n, _ = self.env.step(act_n)

                act_n = add_noise(act_n, self.noise_rate)

                for i in range(self.n):
                    self.agents[i].buffer.add(curr_obs_n[i], act_n[i], next_obs_n[i], reward_n[i], done_n[i])
                curr_obs_n = deepcopy(next_obs_n)

                step += 1
                episode_len += 1
                done = all(done_n)
                terminal = (episode_len >= self.max_episode_len)

                if step % self.update_rate_maddpg == 0:
                    total_critic_loss, total_actor_loss = 0., 0.
                    for i in range(self.n):
                        critic_loss, actor_loss = self.agent_update(i)
                        total_critic_loss += critic_loss
                        total_actor_loss  += actor_loss

                    # log to tensorboard
                    self.writer.add_scalar('Loss/Critic loss', total_critic_loss, step)
                    self.writer.add_scalar('Loss/Actor loss', total_actor_loss, step)

                if step % self.target_update_rate_maddpg:
                    for i in range(self.n):
                        self.agent_target_update(i, self.tau)

                if step % self.update_rate_consensus == 0:
                    total_cons_loss = 0.
                    for i in range(self.args.num_team):
                        total_cons_loss += self.consensus_update(i)
                    self.writer.add_scalar('Consensus loss', total_cons_loss, step)

                if step % self.target_update_rate_consensus:
                    for i in range(self.args.num_team):
                        self.consensus_target_update(i, tau = self.tau)

                if done or terminal:
                    break
            self.noise_rate = max(0.05, self.noise_rate - 1e-4)
            if episode % self.args.evaluate_rate == 0:
                r_avg, r_frd, r_adv = self.evaluate()
                rewards.append(r_avg)
                rewards_frd.append(r_frd)
                rewards_adv.append(r_adv)

                # log to tensorboard
                self.writer.add_scalar('Reward/Avg. reward', r_avg, episode)
                self.writer.add_scalar('Reward/Friend reward', r_frd, episode)
                self.writer.add_scalar('Reward/Adversary reward', r_adv, episode)

                print(f"episode: {episode}\t total reward :{r_avg}\t friend reward :{r_frd}\t adv reward :{r_adv}")

            # if episode % self.args.save_rate == 0:
            #     self.save_models()

    def evaluate(self):
        '''
        Return three rewards: average reward among episodes, average reward among episodes and num_friends, average reward among episodes and num_advesaries,
        '''
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

    def get_minibatch(self, cons = False, group = 0):
        idxs = self.agents[0].buffer.sample_minibatch(self.batch_size)
        if cons:
            agent_idxs = self._sub_agent_idx(group)
            agents = [self.agents[i] for i in agent_idxs]
        else:
            agents = self.agents
            agent_idxs = []
        for idx, agent in enumerate(agents):
            component = agent.buffer.get_minibatch_component(idxs)
            if idx==0:
                minibatch = component
            else:
                minibatch = [ torch.cat((item, component[i]), dim=1) for i, item in enumerate(minibatch) ]
        return minibatch, agent_idxs

    def agent_actor_loss_helper(self, agent_id, minibatch):
        curr_obs_n, act_n, _, _, _ = minibatch
        ### o
        o = curr_obs_n
        ### a
        a = act_n
        agent_curr_obs = self._local_slice(agent_id, curr_obs_n)
        agent_act = self.agents[agent_id].get_action(agent_curr_obs, is_target=False, is_argmax=False)
        a = self._local_replace(agent_id, agent_act, a, is_action=True)
        if self.args.policies[agent_id] == "ddpg":
            o = agent_curr_obs
            a = self._local_slice(agent_id, act_n, is_action=True)
        loss = - self.agents[agent_id].get_q(o, a, is_target=False).mean()
        return loss

    def consensus_actor_loss_helper(self, agent_id, minibatch):
        group = 0 if agent_id < self.args.num_friends else 1
        o_com, a_com = self._reorder_batch(minibatch, agent_id, group)
        loss = - self.cons[group].get_q(o_com, a_com, is_target=False).mean()
        return loss

    def agent_actor_loss(self, agent_id, minibatch):
        l1 = self.agent_actor_loss_helper(agent_id, minibatch)
        if self.args.use_common:
            l2 = self.consensus_actor_loss_helper(agent_id, minibatch)
            loss = l1 * self.args.beta + l2 * (1-self.args.beta)
            return loss
        return l1

    def agent_critic_loss(self, agent_id, minibatch):
        if self.args.policies[agent_id] == "maddpg":
            return self.maddpg_critic_loss(agent_id, minibatch)
        elif self.args.policies[agent_id] == "ddpg":
            return self.ddpg_critic_loss(agent_id, minibatch)

    def agent_update(self, agent_id):
        # get minibatch sample
        minibatch, _ = self.get_minibatch()
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

    def agent_target_update(self, agent_id, tau):
        self.agents[agent_id].target_update(tau, is_actor=True)
        self.agents[agent_id].target_update(tau, is_actor=False)

    def maddpg_critic_loss(self, agent_id, minibatch):
        curr_obs_n, act_n, next_obs_n, reward_n, _ = minibatch
        o = next_obs_n
        a = torch.empty((self.batch_size, self.n * self.act_dim))
        reward_n = reward_n.to(self.args.device)

        for i, agent in enumerate(self.agents):
            curr_obs = self._local_slice(i, curr_obs_n)
            act = agent.get_action(curr_obs, is_target=True, is_argmax=False)
            a = self._local_replace(i, act, a, is_action=True)
        y = self.agents[agent_id].get_q(o, a, is_target=True).squeeze()
        y = self.gamma * y + reward_n[:, agent_id]

        # calculate goal_y
        goal_o = curr_obs_n
        goal_a = act_n
        goal_y = self.agents[agent_id].get_q(goal_o, goal_a, is_target=False).squeeze()
        return F.mse_loss(goal_y, y)

    def ddpg_critic_loss(self, agent_id, minibatch):
        curr_obs_n, act_n, next_obs_n, reward_n, _ = minibatch
        reward_n = reward_n.to(self.args.device)
        o = self._local_slice(agent_id, curr_obs_n)
        a = self._local_slice(agent_id, act_n, is_action = True)
        reward = reward_n[:, agent_id]
        o_next = self._local_slice(agent_id, next_obs_n)
        a_next = self.agents[agent_id].get_action(o_next, is_target=True, is_argmax=False)
        a_next = a_next.to("cpu")

        y = self.agents[agent_id].get_q(o_next, a_next, is_target=True).squeeze()
        y = self.gamma * y + reward

        goal_y = self.agents[agent_id].get_q(o, a, is_target=False).squeeze()

        return F.mse_loss(goal_y, y)

    def consensus_critic_loss(self, group, minibatch, sub_idxs):
        o_com, a_com, o_com_next, r_com, _ = minibatch
        r_com = r_com.to(self.args.device)
        a_com_next = torch.empty((self.batch_size, self.K * self.act_dim))
        for i, idx in enumerate(sub_idxs):
            o = self._local_slice(i, o_com)
            a = self.agents[idx].get_action(o, is_target=True, is_argmax=False)
            a_com_next = self._local_replace(i, a, a_com_next, is_action = True)
        y = self.cons[group].get_q(o_com_next, a_com_next).squeeze()
        y = y * self.gamma + r_com[:, 0]
        y_goal = self.cons[group].get_q(o_com, a_com).squeeze()
        return F.mse_loss(y_goal, y)

    def consensus_update(self, group = 0):
        minibatch, sub_idx = self.get_minibatch(cons=True, group = group)
        critic_loss = self.consensus_critic_loss(group, minibatch, sub_idx)
        self.cons[group].critic_optim.zero_grad()
        critic_loss.backward()
        self.cons[group].critic_optim.step()
        return critic_loss.item()

    def consensus_target_update(self, group, tau):
        self.cons[group].target_update(tau)

    def save_models(self):
        path_root = self.path
        for idx, agent in enumerate(self.agents):
            path = os.path.join(path_root, "agent"+str(idx))
            if not os.path.exists((path)):
                os.makedirs(path)
            torch.save(agent.critic.state_dict(), os.path.join(path, "critic.pt"))
            torch.save(agent.target_critic.state_dict(), os.path.join(path, "t_critic"))
            torch.save(agent.actor.state_dict(), os.path.join(path, "actor"))
            torch.save(agent.target_actor.state_dict(), os.path.join(path, "t_actor"))

    def load_models(self):
        path_root = self.path
        for idx, agent in enumerate(self.agents):
            path = os.path.join(path_root, "agent" + str(idx))
            if not os.path.exists(path):
                continue
            agent.critic.load_state_dict(torch.load(os.path.join(path, "critic")))
            agent.target_critic.load_state_dict(torch.load(os.path.join(path, "t_critic")))
            agent.actor.load_state_dict(torch.load(os.path.join(path, "actor")))
            agent.target_actor.load_state_dict(torch.load(os.path.join(path, "t_actor")))

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

    def _reorder_batch(self, minibatch, agentid, group):
        """
        Given a minibatch, generate a sub minibatch with K agents.
        First K//2 are friends and rest are adv.
        Also, the first agent here correspond to agent id.
        """
        sub_idx = self._sub_agent_idx(group, agentid = agentid)
        o_n, a_n, _, _, _ = minibatch
        a_com = torch.empty((self.batch_size, self.K * self.act_dim))
        o_com = torch.empty((self.batch_size, self.K * self.obs_dim_arr[0]))

        for idx, agentid in enumerate(sub_idx):
            o = self._local_slice(agentid, o_n, is_action=False, is_batch=True)
            a = self._local_slice(agentid, a_n, is_action=True, is_batch=True)
            a_com = self._local_replace(idx, a, a_com, is_action=True)
            o_com = self._local_replace(idx, o, o_com, is_action=False)
        o = self._local_slice(idx = 0, target = o_com, is_action=False, is_batch=True)
        a = self.agents[agentid].get_action(obs = o, is_target=True, is_argmax=False)
        a_com = self._local_replace(0, a, a_com, is_action=True)
        return o_com, a_com

    def _sub_agent_idx(self, group, agentid = -1):
        """
        Given current group, generate a random array with size K. First k/2 elements are index for group(sent as parameter),
         the rest are for the other group. This can make sure current group will always occupy the front part of the index.
        """
        num_group1 = self.K // 2
        num_group2 = self.K - num_group1
        num_friends, num_adv = self.args.num_friends, self.args.num_adversaries
        sub_idx1 = np.random.choice(num_friends, num_group1, replace = False)
        sub_idx2 = np.random.choice(num_adv, num_group2, replace = False)
        if group == 0:
            sub_idx2 += num_friends
            sub_idx = np.append(sub_idx1, sub_idx2)
        else:
            sub_idx1 += num_adv
            sub_idx = np.append(sub_idx2, sub_idx1)
        sub_idx = sub_idx.tolist()
        if agentid != -1:
            if agentid in sub_idx:
                i = sub_idx.index(agentid)
                t = sub_idx[0]
                sub_idx[0] = agentid
                sub_idx[i] = t
            else:
                sub_idx[0] = agentid
        return sub_idx
