"""
entropy-constrained PPO
"""

import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

#Hyperparameters
learning_rate_a = 0.0001
learning_rate_c = 0.005
learning_rate_alpha = 0.005
gamma         = 0.95
lmbda         = 0.95
eps_clip      = 0.1
Range         = 2
K_epoch       = 5
max_grad_norm = 0.5
T_horizon     = 600
traning_step = 10000
seed = 0
mode = "train"   # train or test

env = gym.make('Pendulum-v0')
env.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class ActorNet(nn.Module):
    def __init__(self):
        super(ActorNet, self).__init__()
        self.fc = nn.Linear(3, 128)
        self.fc_ = nn.Linear(128, 128)

        self.mu_head = nn.Linear(128, 1)
        self.sigma_head = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc_(x))

        mu = Range * F.tanh(self.mu_head(x))
        sigma = F.softplus(self.sigma_head(x))
        # print("@@@", mu, sigma)
        return (mu, sigma)


class CriticNet(nn.Module):
    def __init__(self):
        super(CriticNet, self).__init__()
        self.fc = nn.Linear(3, 128)
        self.fc_ = nn.Linear(128, 128)

        self.fc_v = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.fc(x))
        x = F.relu(self.fc_(x))

        v = self.fc_v(x)
        return v


class agent(nn.Module):
    def __init__(self):
        super(agent, self).__init__()
        self.anet = ActorNet().float()
        self.cnet = CriticNet().float()
        self.data = []

        self.alpha = 0.2
        self.target_entropy = -np.prod((1,)).item()
        self.log_alpha = torch.zeros(1, requires_grad=True)
        self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate_alpha)  # 优化温度系数的数值

        self.optimizer_a = optim.Adam(self.anet.parameters(), learning_rate_a)
        self.optimizer_c = optim.Adam(self.cnet.parameters(), learning_rate_c)
      
    def put_data(self, transition):
        self.data.append(transition)
        
    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition
            
            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])
            
        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a
        
    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            mu, sigma = self.anet(s)
            dist = Normal(mu, sigma)
            pi_a = dist.log_prob(a).exp()
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            td_target = r + gamma * (self.cnet(s_prime)-self.alpha * torch.log(pi_a)) * done_mask
            delta = td_target - self.cnet(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss_a = self.alpha * torch.log(pi_a)-torch.min(surr1, surr2)

            self.optimizer_a.zero_grad()
            loss_a.mean().backward()
            nn.utils.clip_grad_norm_(self.anet.parameters(), max_grad_norm)
            self.optimizer_a.step()

            loss_c = F.smooth_l1_loss(self.cnet(s), td_target.detach())
            self.optimizer_c.zero_grad()
            loss_c.mean().backward()
            nn.utils.clip_grad_norm_(self.cnet.parameters(), max_grad_norm)
            self.optimizer_c.step()

            alpha_loss = -(self.log_alpha * (torch.log(pi_a) + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            self.alpha = self.log_alpha.exp()
            # print("###alpha_loss&self.alpha###", alpha_loss, self.alpha)

    def select_action(self, s, mode):
        mu, sigma = self.anet(torch.from_numpy(s).float())
        if mode == "test":
            dist = Normal(mu, sigma)
            action = mu
            action = action.clamp(-Range, Range)
            prob = dist.log_prob(action).exp()
        else:
            dist = Normal(mu, sigma)
            action = dist.rsample()   # reparameterization trick (mean + std * N(0,1))
            prob = dist.log_prob(action).exp()
            action = action.clamp(-Range, Range)
        return action.item(), prob.item()

    def load_model(self):
        self.anet = torch.load('/home/hxk/Code/Highway_SACPPO_v_ITSC/model_v_new/v_anet3080.pkl')
        self.cnet = torch.load('/home/hxk/Code/Highway_SACPPO_v_ITSC/model_v_new/v_cnet3080.pkl')

    def save_model(self):
        torch.save(self.anet, '/home/hxk/Code/Highway_SACPPO_v_ITSC/model_v_new/v_anet3080.pkl')
        torch.save(self.cnet, '/home/hxk/Code/Highway_SACPPO_v_ITSC/model_v_new/v_cnet3080.pkl')


def main():
    model = agent()
    score = 0.0
    episode = []
    reward = []
    print_interval = 20

    for n_epi in range(traning_step):
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                a, prob = model.select_action(s, mode)
                s_prime, r, done, info = env.step([a])

                model.put_data((s, a, (r+8)/8, s_prime, prob, done))
                s = s_prime

                score += r
                if done:
                    break

                if (t+1) % 100 == 0:
                    model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.1f}".format(n_epi, score/print_interval))
            episode.append(n_epi)
            reward.append(score/print_interval)
            score = 0.0

    # model.save_model()
    # print("###The model is saved###")

    plt.plot(episode, reward)
    plt.xlabel('episode')
    plt.ylabel('reward_ecppo')
    plt.show()

    env.close()

if __name__ == '__main__':
    main()
