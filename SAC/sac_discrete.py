import gym
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import matplotlib.pyplot as plt

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Configurations
parser = argparse.ArgumentParser(description='RL algorithms with PyTorch in Pendulum environment')
parser.add_argument('--env', type=str, default='CartPole-v0', help='CartPole environment')
parser.add_argument('--seed', type=int, default=0, help='seed for random number generators')
parser.add_argument('--training_eps', type=int, default=200, help='training episode number')
parser.add_argument('--evaluation_eps', type=int, default=10, help='evaluation episode number')
parser.add_argument('--max_step', type=int, default=200, help='max episode step')
parser.add_argument('--state_dim', type=int, default=4, help='state dimension')
parser.add_argument('--action_dim', type=int, default=1, help='action dimension')
parser.add_argument('--action_numb', type=int, default=2, help='action number')
args = parser.parse_args()


class ActorNet(nn.Module):
    def __init__(self, state_dim, action_numb):
        super(ActorNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.pi = nn.Linear(128, action_numb)

    def forward(self, x, softmax_dim=0):
        x = F.relu(self.fc1(x))
        x = self.pi(x)

        prob = F.softmax(x, dim=softmax_dim)
        return prob


class CriticNet(nn.Module):
    def __init__(self, state_dim, action_numb):
        super(CriticNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_numb)

    def forward(self, s):
        x = F.relu(self.fc1(s))
        q = self.fc2(x)
        return q


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
    An implementation of Automatic entropy adjustment SAC (ASAC)
    """
    def __init__(self,
                 env,
                 args,
                 state_dim,
                 action_dim,
                 action_numb,
                 steps=0,
                 gamma=0.99,
                 alpha=0.2,
                 automatic_entropy_tuning=False,
                 buffer_size=int(1000000),
                 batch_size=128,  # 64
                 actor_lr=1e-4,
                 qf_lr=1e-3,
                 alpha_lr=1e-4,
                 ):
        super(Agent, self).__init__()

        self.env = env
        self.args = args
        self.obs_dim = state_dim
        self.act_dim = action_dim
        self.act_num = action_numb
        self.steps = steps
        self.gamma = gamma
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.actor_lr = actor_lr
        self.qf_lr = qf_lr
        self.alpha_lr = alpha_lr

        # Main network
        self.actor = ActorNet(self.obs_dim, self.act_num).to(device)
        self.qf1 = CriticNet(self.obs_dim, self.act_num).to(device)
        self.qf2 = CriticNet(self.obs_dim, self.act_num).to(device)
        # Target network
        self.qf1_target = CriticNet(self.obs_dim, self.act_num).to(device)
        self.qf2_target = CriticNet(self.obs_dim, self.act_num).to(device)

        # Initialize target parameters to match main parameters
        self.qf1_target.load_state_dict(self.qf1.state_dict())
        self.qf2_target.load_state_dict(self.qf2.state_dict())

        # Create optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)  
        self.qf1_optimizer = optim.Adam(self.qf1.parameters(), lr=self.qf_lr)
        self.qf2_optimizer = optim.Adam(self.qf2.parameters(), lr=self.qf_lr)

        # Experience buffer
        self.replay_buffer = ReplayBuffer(self.obs_dim, self.act_dim, self.buffer_size)

        # If automatic entropy tuning is True,
        if self.automatic_entropy_tuning:
            self.target_entropy = -np.prod((self.act_dim,)).item()
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=self.alpha_lr)  

    def train_model(self):
        batch = self.replay_buffer.sample(self.batch_size)
        obs1 = batch['obs1']
        obs2 = batch['obs2']
        acts = batch['acts']
        rews = batch['rews']
        done = batch['done']

        # Prediction π(s), logπ(s), π(s'), logπ(s'), Q1(s,a), Q2(s,a)
        a, pi_a, log_pi, prob = self.select_action_batch(obs1)
        a_next, next_pi, next_log_pi, next_prob = self.select_action_batch(obs2)
        log_pi = log_pi.squeeze(1)
        next_log_pi = next_log_pi.squeeze(1)

        q1 = self.qf1(obs1).gather(1, acts.long()).squeeze(1)
        q2 = self.qf2(obs1).gather(1, acts.long()).squeeze(1)
        q1_pi_next = self.qf1_target(obs2)
        q2_pi_next = self.qf2_target(obs2)

        # Min Double-Q: min(Q1(s,π(s)), Q2(s,π(s))), min(Q1‾(s',π(s')), Q2‾(s',π(s')))
        min_q_next_pi = torch.min(q1_pi_next, q2_pi_next).to(device)

        # Targets for Q and V regression
        v_backup = (next_prob*min_q_next_pi).sum(dim=-1) - self.alpha * next_log_pi
        q_backup = rews + self.gamma * (1 - done) * v_backup
        q_backup.to(device)

        # Soft actor losses
        with torch.no_grad():
            q1_pi = self.qf1(obs1)
            q2_pi = self.qf2(obs1)
            min_q_pi = torch.min(q1_pi, q2_pi).to(device)
        actor_loss = (self.alpha * log_pi - (prob*min_q_pi).sum(dim=-1)).mean()

        # Soft critic losses
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
            alpha_loss = -(self.log_alpha * (log_pi.detach() + self.target_entropy)).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.detach().exp()

        # Polyak averaging for target parameter
        self.soft_target_update(self.qf1, self.qf1_target)
        self.soft_target_update(self.qf2, self.qf2_target)

    def soft_target_update(self, main, target, tau=0.005):
        for main_param, target_param in zip(main.parameters(), target.parameters()):
            target_param.data.copy_(tau * main_param.data + (1.0 - tau) * target_param.data)

    def select_action_batch(self, state):
        state = torch.FloatTensor(state).to(device)
        prob = self.actor(state, softmax_dim=1)
        m = Categorical(prob)
        action = m.sample().reshape([-1, 1])
        pi_a = prob.gather(1, action)
        log_pi = torch.log(pi_a)
        return action, pi_a, log_pi, prob

    def select_action_single(self, state):
        state = torch.FloatTensor(state).to(device)
        prob = self.actor(state, softmax_dim=0)
        m = Categorical(prob)
        action = m.sample()
        a = action.detach().cpu().numpy()
        return a

    def run(self, max_step):
        step_number = 0
        total_reward = 0.

        obs = self.env.reset()
        done = False

        # Keep interacting until agent reaches a terminal state.
        while not (done or step_number == max_step):
            self.steps += 1

            # Collect experience (s, a, r, s') using some policy
            action = self.select_action_single(torch.Tensor(obs).to(device))
            next_obs, reward, done, _ = self.env.step(action)

            # Add experience to replay buffer
            self.replay_buffer.add(obs, action, reward, next_obs, done)

            # Start training when the number of experience is greater than batch size
            if self.steps > self.batch_size:
                self.train_model()

            total_reward += reward
            step_number += 1
            obs = next_obs

        return step_number, total_reward

    def train(self, mode: bool = True) -> "ASAC":
        self.actor.train(mode)
        self.qf1.train(mode)
        self.qf2.train(mode)
        return self


def main():
    """Main."""
    # Set environment
    env = gym.make(args.env)

    # Set a random seed
    env.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Initialize environment
    print('State dimension:', args.state_dim)
    print('Action dimension:', args.action_dim)
    print("Action numb", args.action_numb)
    start_time = time.time()

    # Create an agent
    agent = Agent(env, args, args.state_dim, args.action_dim, args.action_numb, automatic_entropy_tuning=True)

    train_num_steps = 0
    train_sum_returns = 0.
    train_num_episodes = 0
    train_sum_returns_list = []
    episodes_list = []

    # Runs a full experiment, spread over multiple training episodes
    for episode in range(1, args.training_eps + 1):
        # Perform the training phase, during which the agent learns
        # Run one episode
        train_step_length, train_episode_return = agent.run(args.max_step)

        train_num_steps += train_step_length
        train_sum_returns += train_episode_return
        train_num_episodes += 1

        # Perform the evaluation phase -- no learning
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
