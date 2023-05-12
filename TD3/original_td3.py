import os
import gym
import time
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
parser.add_argument('--action_limit', type=int, default=2, help='action limit')
parser.add_argument('--hidden_sizes', type=int, default=128, help='hidden_sizes')   # 32
args = parser.parse_args()

# Set a random seed
np.random.seed(args.seed)
torch.manual_seed(args.seed)

action_limit = args.action_limit
action_dim = 1
state_dim = 3


class ActorNet(nn.Module):
    def __init__(self, state_dim):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, args.hidden_sizes)
        self.fc2 = nn.Linear(args.hidden_sizes, args.hidden_sizes)
        self.mu_head = nn.Linear(args.hidden_sizes, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        return self.mu_head(x)


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, args.hidden_sizes)
        self.fc2 = nn.Linear(args.hidden_sizes, args.hidden_sizes)
        self.fc3 = nn.Linear(args.hidden_sizes, 1)

    def forward(self, s, a):
        s = s.reshape(-1, state_dim)
        a = a.reshape(-1, action_dim)
        x = torch.cat((s, a), -1)  # combination s and a
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class ReplayBuffer(object):
    """
    A simple FIFO experience replay buffer for agents.
    """

    def __init__(self, obs_dim, act_dim, size):
        self.obs1_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.obs2_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size, act_dim], dtype=np.float32)
        self.rews_buf = np.zeros(size, dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def add(self, obs, act, rew, next_obs, done):
        self.obs1_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample(self, batch_size=64):
        idxs = np.random.randint(0, self.size, size=batch_size)
        return dict(obs1=torch.Tensor(self.obs1_buf[idxs]).to(device),
                    obs2=torch.Tensor(self.obs2_buf[idxs]).to(device),
                    acts=torch.Tensor(self.acts_buf[idxs]).to(device),
                    rews=torch.Tensor(self.rews_buf[idxs]).to(device),
                    done=torch.Tensor(self.done_buf[idxs]).to(device))


class OrnsteinUhlenbeckNoise:
    def __init__(self, mu):
        self.theta, self.dt, self.sigma = 0.1, 0.01, 0.1
        self.mu = mu
        self.x_prev = np.zeros_like(self.mu)

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x


class Agent():
    """
    An implementation of Soft Actor-Critic (SAC), Automatic entropy adjustment SAC (ASAC)
    """

    def __init__(self,
                 state_dim=state_dim,
                 action_dim=action_dim,
                 action_limit=action_limit,
                 steps=0,
                 gamma=0.99,
                 automatic_entropy_tuning=False,
                 hidden_sizes=(128, 128),
                 buffer_size=int(1e6),
                 batch_size=128,  # 64
                 actor_lr=1e-3,
                 qf_lr=1e-3,
                 ):
        super(Agent, self).__init__()

        self.obs_dim = state_dim
        self.act_dim = action_dim
        self.act_limit = action_limit
        self.steps = steps
        self.gamma = gamma
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.hidden_sizes = hidden_sizes
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.qf_lr = qf_lr
        self.train_times = 0

        # Main network
        self.actor = ActorNet(state_dim).to(device)
        self.qf1 = CriticNet(state_dim, action_dim).to(device)
        self.qf2 = CriticNet(state_dim, action_dim).to(device)

        # Target network
        self.actor_target = ActorNet(state_dim).to(device)
        self.qf1_target = CriticNet(state_dim, action_dim).to(device)
        self.qf2_target = CriticNet(state_dim, action_dim).to(device)

        # Initialize target parameters to match main parameters
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)  # 优化网络的参数的数值
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=self.qf_lr)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size)

        # action noise
        self.ou_noise = OrnsteinUhlenbeckNoise(mu=np.zeros(1))

    def train_model(self):
        batch = self.replay_buffer.sample(self.batch_size)
        obs1 = batch['obs1']
        obs2 = batch['obs2']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']

        if 0:  # Check shape of experiences
            print("obs1", obs1.shape)
            print("obs2", obs2.shape)
            print("acts", acts.shape)
            print("rews", rews.shape)
            print("done", done.shape)

        # Prediction π(s), logπ(s), π(s'), logπ(s'), Q1(s,a), Q2(s,a)
        action, pi = self.select_action(obs1)

        mu_next = self.actor_target(torch.FloatTensor(obs2).to(device))
        a_next = mu_next + self.ou_noise()[0]
        next_pi = torch.tanh(a_next)

        q1 = self.qf1(obs1, acts).squeeze(1)
        q2 = self.qf2(obs1, acts).squeeze(1)

        # Min Double-Q: min(Q1‾(s',π(s')), Q2‾(s',π(s')))
        q_pi = self.qf1(obs1, pi).squeeze(1).to(device)
        min_q_next_pi = torch.min(self.qf1_target(obs2, next_pi), self.qf2_target(obs2, next_pi)).squeeze(1).to(device)

        # Targets for Q and V regression
        v_backup = min_q_next_pi
        q_backup = rews + self.gamma * (1 - done) * v_backup
        q_backup.to(device)

        if 0:  # Check shape of prediction and target
            print("pi", pi.shape)
            print("next_pi", next_pi.shape)
            print("q1", q1.shape)
            print("q2", q2.shape)
            print("q_pi", q_pi.shape)
            print("min_q_next_pi", min_q_next_pi.shape)
            print("q_backup", q_backup.shape)

        # Soft actor-critic losses
        actor_loss = (- q_pi).mean()
        qf1_loss = F.mse_loss(q1, q_backup.detach())
        qf2_loss = F.mse_loss(q2, q_backup.detach())

        # Update two Q network parameter
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        if self.train_times % 2 == 0:
            # Update actor network parameter
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Polyak averaging for target parameter
            self.soft_target_update(self.actor, self.actor_target)
            self.soft_target_update(self.qf1, self.qf1_target)
            self.soft_target_update(self.qf2, self.qf2_target)

        self.train_times += 1

    def soft_target_update(self, main, target, tau=0.005):
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def select_action(self, state, mode='train'):
        state = torch.FloatTensor(state).to(device)
        mu = self.actor(state)
        # print("$$$mu$$$", mu[0])

        if mode == 'train':
            a = mu + self.ou_noise()[0]
            a = action_limit*torch.tanh(a).detach().cpu().numpy()
        else:
            a = action_limit*torch.tanh(mu)
        return a, torch.tanh(mu)

    def train(self, mode: bool = True) -> "ASAC":
        self.actor.train(mode)
        self.qf1.train(mode)
        return self

    def load_model(self, model_name):
        name = "./model_v/policy_v%d" % model_name
        self.actor = torch.load("{}.pkl".format(name))

    def save_model(self, model_name):
        name = "./model_v/policy_v%d" % model_name
        torch.save(self.actor, "{}.pkl".format(name))


def main():
    env = gym.make('Pendulum-v0')
    model = Agent()

    total_reward = []
    episode = []
    score = 0.0
    print_interval = 10

    for n_epi in range(1000):
        s = env.reset()
        for t in range(200):
            a, mu = model.select_action(s)
            # print("a", a.item())
            s_prime, r, done, info = env.step([a.item()])
            model.replay_buffer.add(s, a/action_limit, r/100.0, s_prime, done)
            s = s_prime

            if n_epi > 0:
                model.train_model()

            score += r
            if done:
                break

        if n_epi % print_interval == 0 and n_epi > 0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            episode.append(n_epi)
            total_reward.append(score / print_interval)
            score = 0.0

    plt.plot(episode, total_reward)
    plt.xlabel('episode')
    plt.ylabel('total_reward')
    plt.show()

    env.close()


if __name__ == '__main__':
    main()




