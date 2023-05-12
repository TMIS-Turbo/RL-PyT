import gym
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

#Hyperparameters
learning_rate_a = 0.00005
learning_rate_c = 0.005
gamma         = 0.99
lmbda         = 0.97
eps_clip      = 0.2
Range         = 2
K_epoch       = 10
T_horizon     = 100
traning_step = 5000
max_grad_norm = 1

# Set environment
env = gym.make('Pendulum-v0')

# Set a random seed
env.seed(0)
np.random.seed(0)
torch.manual_seed(0)


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
            td_target = r + gamma * self.cnet(s_prime) * done_mask
            delta = td_target - self.cnet(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            mu, sigma = self.anet(s)
            dist = Normal(mu, sigma)
            pi_a = dist.log_prob(a).exp()
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))
            # print("$$$ratio$$$", ratio)

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss_a = -torch.min(surr1, surr2)

            self.optimizer_a.zero_grad()
            loss_a.mean().backward()
            nn.utils.clip_grad_norm_(self.anet.parameters(), max_grad_norm)
            self.optimizer_a.step()

            loss_c = F.smooth_l1_loss(self.cnet(s), td_target.detach())
            self.optimizer_c.zero_grad()
            loss_c.mean().backward()
            nn.utils.clip_grad_norm_(self.cnet.parameters(), max_grad_norm)
            self.optimizer_c.step()

    def select_action(self, s):
        mu, sigma = self.anet(torch.from_numpy(s).float())
        dist = Normal(mu, sigma)
        action = dist.sample()
        #print("###", action)
        prob = dist.log_prob(action).exp()
        action = action.clamp(-2.0, 2.0)

        action = action.item()
        prob = prob.item()
        return action, prob

    def load_model(self):
        self.anet = torch.load('/home/ppo_anet.pkl')
        self.cnet = torch.load('/home/ppo_cnet.pkl')

    def save_model(self):
        torch.save(self.anet, '/home/hxk/Code_Baseline/My_Baselines0209/model/ppo_anet.pkl')
        torch.save(self.cnet, '/home/hxk/Code_Baseline/My_Baselines0209/model/ppo_cnet.pkl')

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
                a, prob = model.select_action(s)
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
    plt.ylabel('reward_ppo')
    plt.show()

    env.close()

if __name__ == '__main__':
    main()