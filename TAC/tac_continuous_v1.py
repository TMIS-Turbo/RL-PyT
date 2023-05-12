import os
import gym
import time
import argparse
import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--env', type=str, default='Pendulum-v0', help='pendulum environment')
parser.add_argument('--algo', type=str, default='atac', help='select an algorithm among tac, atac')
parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
parser.add_argument('--training_eps', type=int, default=500, help='training episode number')
parser.add_argument('--evaluation_eps', type=int, default=10, help='evaluation number per training')
parser.add_argument('--max_step', type=int, default=200, help='max episode step')
parser.add_argument('--threshold_return', type=int, default=-215, help='solved requirement for success in given environment')
parser.add_argument('--state_dim', type=int, default=3, help='state dimension')
parser.add_argument('--action_dim', type=int, default=1, help='action dimension')
parser.add_argument('--action_limit', type=int, default=2, help='action limit')
args = parser.parse_args()

# Set environment
env = gym.make(args.env)

# Set a random seed
env.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

state_dim = args.state_dim
action_dim = args.action_dim
action_limit = args.action_limit


class ActorNet(nn.Module):
    def __init__(self, state_dim, min_log_std=-20, max_log_std=20):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu_head = nn.Linear(256, 1)
        self.log_std_head = nn.Linear(256, 1)

        self.min_log_std = min_log_std
        self.max_log_std = max_log_std

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        mu = self.mu_head(x)
        log_std_head = F.relu(self.log_std_head(x))
        log_std_head = torch.clamp(log_std_head, self.min_log_std, self.max_log_std)

        return mu, log_std_head


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

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


class Agent():
    """
    An implementation of Soft Actor-Critic (SAC), Automatic entropy adjustment SAC (ASAC)
    """

    def __init__(self,
                 env,
                 args,
                 state_dim,
                 action_dim,
                 action_limit,
                 steps=0,
                 gamma=0.99,
                 alpha=0.2,
                 entropic_index=0.5,
                 automatic_entropy_tuning=False,
                 hidden_sizes=(128, 128),
                 buffer_size=int(1e6),
                 batch_size=128,  # 64
                 actor_lr=3e-4,
                 qf_lr=1e-3,
                 alpha_lr=3e-4,
                 eval_mode=False,
                 actor_losses=list(),
                 qf1_losses=list(),
                 qf2_losses=list(),
                 alpha_losses=list(),
                 logger=dict(),
                 ):
        super(Agent, self).__init__()

        self.env = env
        self.args = args
        self.obs_dim = state_dim
        self.act_dim = action_dim
        self.act_limit = action_limit
        self.steps = steps
        self.gamma = gamma
        self.alpha = alpha
        self.q = entropic_index
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.hidden_sizes = hidden_sizes
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.qf_lr = qf_lr
        self.alpha_lr = alpha_lr
        self.eval_mode = eval_mode
        self.actor_losses = actor_losses
        self.qf1_losses = qf1_losses
        self.qf2_losses = qf2_losses
        self.alpha_losses = alpha_losses
        self.logger = logger

        # Main network
        self.actor = ActorNet(state_dim).to(device)
        self.qf1 = CriticNet(state_dim, action_dim).to(device)
        self.qf2 = CriticNet(state_dim, action_dim).to(device)
        # Target network
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

        # If automatic entropy tuning is True,
        # initialize a target entropy, a log alpha and an alpha optimizer
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod((action_dim,)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)  # 优化温度系数的数值

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
        action, pi, log_pi = self.select_action(obs1)
        _, next_pi, next_log_pi = self.select_action(obs2)
        q1 = self.qf1(obs1, acts).squeeze(1)
        q2 = self.qf2(obs1, acts).squeeze(1)

        # Min Double-Q: min(Q1(s,π(s)), Q2(s,π(s))), min(Q1‾(s',π(s')), Q2‾(s',π(s')))
        min_q_pi = torch.min(self.qf1(obs1, pi), self.qf2(obs1, pi)).squeeze(1).to(device)
        min_q_next_pi = torch.min(self.qf1_target(obs2, next_pi), self.qf2_target(obs2, next_pi)).squeeze(1).to(device)

        # Targets for Q and V regression
        v_backup = min_q_next_pi - self.alpha * next_log_pi
        q_backup = rews + self.gamma * (1 - done) * v_backup
        q_backup.to(device)

        if 0:  # Check shape of prediction and target
            print("action", action)
            print("pi", pi.shape)
            print("log_pi", log_pi.shape)
            print("q1", q1.shape)
            print("q2", q2.shape)
            print("min_q_pi", min_q_pi.shape)
            print("min_q_next_pi", min_q_next_pi.shape)
            print("q_backup", q_backup.shape)

        # Soft actor-critic losses
        actor_loss = (self.alpha * log_pi - min_q_pi).mean()
        qf1_loss = F.mse_loss(q1, q_backup.detach())
        qf2_loss = F.mse_loss(q2, q_backup.detach())

        # Update two Q network parameter
        self.qf1_optimizer.zero_grad()
        qf1_loss.backward()
        self.qf1_optimizer.step()

        self.qf2_optimizer.zero_grad()
        qf2_loss.backward()
        self.qf2_optimizer.step()

        # Update actor network parameter
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # If automatic entropy tuning is True, update alpha
        if self.automatic_entropy_tuning:
            alpha_loss = -(self.log_alpha * (log_pi + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()

            # Save alpha loss
            self.alpha_losses.append(alpha_loss.item())

        # Polyak averaging for target parameter
        self.soft_target_update(self.qf1, self.qf1_target)
        self.soft_target_update(self.qf2, self.qf2_target)

        # Save losses
        self.actor_losses.append(actor_loss.item())
        self.qf1_losses.append(qf1_loss.item())
        self.qf2_losses.append(qf2_loss.item())

    def soft_target_update(self, main, target, tau=0.005):
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def tsallis_entropy_log_q(self, x, q):
        safe_x = torch.max(x, torch.Tensor([1e-6]).to(device))

        if q == 1:
            log_q_x = torch.log(safe_x)
        else:
            log_q_x = (safe_x.pow(q-1)-1)/(q-1)
        return log_q_x.sum(dim=-1)

    def select_action(self, state):
        state = torch.FloatTensor(state).to(device)
        mu, log_sigma = self.actor(state)
        sigma = torch.exp(log_sigma)
        # print("mu", mu.shape)
        # print("log_sigma", log_sigma.shape)

        dist = Normal(mu, sigma)
        z = dist.rsample()   # reparameterization trick (mean + std * N(0,1))
        pi_tensor = torch.tanh(z)

        log_pi = dist.log_prob(z)
        log_pi -= torch.log(1 - pi_tensor.pow(2) + 1e-6)
        exp_log_pi = torch.exp(log_pi)
        log_pi = self.tsallis_entropy_log_q(exp_log_pi, self.q)
        # print("###action&z###", pi_tensor, z)

        action = action_limit*torch.tanh(z).detach().cpu().numpy()
        return action, pi_tensor, log_pi

    def train(self, mode: bool = True) -> "ASAC":
        self.actor.train(mode)
        self.qf1.train(mode)
        self.qf2.train(mode)
        return self


def main():
    """Main."""
    # Initialize environment
    print('State dimension:', state_dim)
    print('Action dimension:', action_dim)
    start_time = time.time()

    # Create an agent
    agent = Agent(env, args, state_dim, action_dim, action_limit, automatic_entropy_tuning=True)

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0
    train_sum_returns_list = []
    episodes_list = []

    # Runs a full experiment, spread over multiple training episodes
    for episode in range(1, args.training_eps + 1):
        # Perform the training phase, during which the agent learns
        agent.eval_mode = False

        # Run one episode
        step_number = 0
        total_reward = 0.

        obs = agent.env.reset()
        done = False

        # Keep interacting until agent reaches a terminal state.
        while not (done or step_number == args.max_step):
            agent.steps += 1

            if agent.eval_mode:
                action, _, _ = agent.select_action(obs)
                next_obs, reward, done, _ = agent.env.step(action)
            else:
                # Collect experience (s, a, r, s') using some policy
                action, _, _ = agent.select_action(obs)
                next_obs, reward, done, _ = agent.env.step(action)
                # print("###", obs, action, reward, next_obs, done)

                # Add experience to replay buffer
                agent.replay_buffer.add(obs, action, reward, next_obs, done)

                # Start training when the number of experience is greater than batch size
                if agent.steps > agent.batch_size:
                    agent.train_model()

            total_reward += reward
            step_number += 1
            obs = next_obs

        train_num_steps += step_number
        train_sum_returns += total_reward
        train_num_episodes += 1

        if episode % args.evaluation_eps == 0:
            train_average_return = train_sum_returns / train_num_episodes if train_num_episodes > 0 else 0.0
            train_sum_returns_list.append(train_average_return)
            episodes_list.append(episode)

            print('---------------------------------------')
            print('Steps:', train_num_steps)
            print('Episodes:', episode)
            print('TestSteps:', train_num_episodes)
            print('AverageReturn:', round(train_average_return, 2))
            print('Time:', int(time.time() - start_time))
            print('---------------------------------------')

            train_sum_returns = 0
            train_num_episodes = 0

    plt.plot(episodes_list, train_sum_returns_list)
    plt.xlabel('Episodes')
    plt.ylabel('AverageReturns')
    plt.show()


if __name__ == '__main__':
    main()