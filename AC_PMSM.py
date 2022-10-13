#%%
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from nb_env import PmsmEnv
import pandas as pd
from torch.nn import functional as F
import random
import matplotlib.pyplot as plt
import csv
import os


MEMORY_CAPACITY = 200
HIDDEN_SIZE = 256
BATCH_SIZE = 64
GAMMA = 0.7
EPISODES = 100
EP_STEPS = 20000
TAU = 0.5
LR = 1e-5
a_dim = 1
s_dim = 2
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")

def seed_everything(seed: int):

    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
seed_everything(2)

def atanh(x,a):
   x =  torch.where(abs(x)>a,torch.sign(x)*a,torch.sign(x)*pow(abs(x),-0.5))
   return x
class Actor(nn.Module):
    def __init__(self, s_dim, a_dim, std=0.0):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(s_dim, HIDDEN_SIZE)
        self.fc1.weight.data.normal_(0, 0.01) # initialization of FC1
        self.out = nn.Linear(HIDDEN_SIZE, a_dim)
        self.out.weight.data.normal_(0, 0.01)        
    def forward(self,x):
        x = self.fc1(x)
        x = F.leaky_relu(x,0.5)
        x = self.out(x)
        x = torch.tanh(x)*50
        return x

class Critic(nn.Module):
    def __init__(self, s_dim,a_dim):
        super(Critic, self).__init__()  
        self.fcs = nn.Linear(s_dim,30)
        self.fcs.weight.data.normal_(0,0.1) 
        self.fca = nn.Linear(a_dim, 30) 
        self.fca.weight.data.normal_(0, 0.1) 
        self.out = nn.Linear(30, 1) 
        self.out.weight.data.normal_(0, 0.1)        
    def forward(self,s,a):
        x = self.fcs(s)
        y = self.fca(a)
        actions_value = self.out(F.relu(x+y))
        return actions_value
#%%
class DDPG(object):
    def __init__(self,a_dim,s_dim):
        self.a_dim, self.s_dim = a_dim,s_dim
        self.memory = torch.zeros((MEMORY_CAPACITY,s_dim*2+a_dim+1),dtype=torch.float32)
        self.pointer = 0
        self.actor_eval = Actor(s_dim, a_dim).to(device)
        self.actor_target = Actor(s_dim, a_dim).to(device)
        self.critic_eval = Critic(s_dim, a_dim).to(device)
        self.critic_target = Critic(s_dim, a_dim).to(device)      
        self.actor_optim =  optim.Adam(self.actor_eval.parameters(),lr=LR)
        self.critic_optim =  optim.Adam(self.critic_eval.parameters(),lr=LR)
        self.loss_func = nn.MSELoss()
        
    def store_transition(self,s,a,r,s_):
        transition = torch.hstack((s,a,r,s_))
        index = self.pointer%MEMORY_CAPACITY
        self.memory[index,:]=transition
        self.pointer += 1
    def choose_action(self,s):
        return self.actor_eval(s).detach()
    def learn(self):
        for x in self.actor_target.state_dict().keys():
            eval('self.actor_target.' + x + '.data.mul_((1-TAU))')
            eval('self.actor_target.' + x + '.data.add_(TAU*self.actor_eval.' + x + '.data)')
        for x in self.critic_target.state_dict().keys():
            eval('self.critic_target.' + x + '.data.mul_((1-TAU))')
            eval('self.critic_target.' + x + '.data.add_(TAU*self.critic_eval.' + x + '.data)')
        indices = np.random.choice(MEMORY_CAPACITY,size=BATCH_SIZE)
        batch_trans = self.memory[indices,:].to(device)
        batch_s = batch_trans[:, :self.s_dim]
        batch_a = batch_trans[:, self.s_dim:self.s_dim + self.a_dim]
        batch_r = batch_trans[:, -self.s_dim - 1: -self.s_dim]
        batch_s_ = batch_trans[:, -self.s_dim:]
        a = self.actor_eval(batch_s)
        a.retain_grad()
        q = self.critic_eval(batch_s, a)
        q.retain_grad()
        actor_loss = -torch.mean(q)
        self.actor_optim.zero_grad()
        actor_loss.retain_grad()
        actor_loss.backward()
        self.actor_optim.step()    
        a_target = self.actor_target(batch_s_)
        q_tmp = self.critic_target(batch_s_, a_target)
        q_target = batch_r + GAMMA * q_tmp
        # compute the current q value and the loss
        q_eval = self.critic_eval(batch_s, batch_a)
        td_error = self.loss_func(q_target, q_eval)/2
        # optimize the loss of critic network
        self.critic_optim.zero_grad()
        td_error.backward()
        self.critic_optim.step()  
        return actor_loss.detach(), td_error.detach()


#%%
a_dim = 1
s_dim = 2

total_steps = 100
n_p =4 
sif= 0.0152
J = 11*1e-4
B = 3.5*1e-5
L = 1.2
Rs = 0.4
a = n_p*sif/J
b = 1/L
g_1 = -B/J
m = -Rs/L
n = -n_p*sif/L
f = torch.Tensor(np.array([[g_1,a],[n,m]])).to(device)
g = torch.Tensor(np.array([[0],[b]])).to(device)

#%%
file = open("D:\PythonProject\RL/record_newth_1.csv","w")
file_ac = open("D:\PythonProject\RL/record_loss_newth_1.csv","w")
writer = csv.writer(file)
writer_ac = csv.writer(file_ac)
writer.writerow(['i','w','iq','w-w_hat'])
ddpg = DDPG(a_dim,s_dim)
df = pd.read_excel("D:/PythonProject/RL/data.xls").to_numpy()
x_0 = torch.Tensor(df).to(device)
for i in range(3):
    i1 = random.randint(3,len(df)-1)
    kuaixie = []
    dianliu = []
    diff = []
    actor_loss = []
    critic_loss = []
    w = x_0[i1,0]
    iq = x_0[i1,1] 
    env = PmsmEnv(f=f,g=g,C_w = 1,w_t_hat=100,C_u = 0.001,dt=0.01,w=w,iq=iq)
    ep_r = 0
    for j in range(EP_STEPS):
        s = env.x
        a = ddpg.choose_action(env.x)
        r,s_,done,ut = env.step(a)
        r = r.reshape(1,1)
        ddpg.store_transition(s,a,r,s_.T)
        if ddpg.pointer > MEMORY_CAPACITY:
            al,cl=ddpg.learn()
            actor_loss.append(al.cpu().item())
            critic_loss.append(cl.cpu().item())
            writer_ac.writerow([i,al.detach().cpu().item(),cl.detach().cpu().item()])
            if j%1000 == 0:
                print(s)
        
        diff.append((s[0,0]-env.w_t_hat).detach().cpu().item())
        kuaixie.append(s[0,0].detach().cpu().item())
        dianliu.append(s[0,1].detach().cpu().item())
        writer.writerow([i,kuaixie[j],dianliu[j],diff[j]])
        ep_r += r
        if j == EP_STEPS - 1:
            print('Episode: ', i, ' Reward: %i' % (ep_r))
            # plt.plot(actor_loss)
            # plt.show()
            # plt.plot(critic_loss)
            # plt.show()
file.close()
file_ac.close()



file = open("record_test2.csv", "w")
writer = csv.writer(file)
zhuansu = [500, 1000, 1500]
file_ut = open("record_ut2.csv", "w")
writer_ut = csv.writer(file_ut)
writer_ut.writerow(["kind of w_t", "u_t"])
for time in range(len(zhuansu)):
    for i in range(4, 5):
        w = x_0[i+2, 0]
        iq = x_0[i+2, 1]
        w_track = []
        iq_track = []
        err_track = []
        env1 = PmsmEnv(f=f, g=g, C_w=1, w_t_hat=zhuansu[time], C_u=0.001, dt=0.01, w=w, iq=iq)
        ep_r = 0
        for j in range(EP_STEPS):
            s = env1.x
            a = ddpg.choose_action(env1.x)
            r, s_, done, ut = env1.step(a)
            writer_ut.writerow([zhuansu[time], ut.item()])
            err_track.append((s[0, 0] - env1.w_t_hat).detach().cpu().item())
            w_track.append(s[0, 0].detach().cpu().item())
            iq_track.append(s[0, 1].detach().cpu().item())
            ep_r += r
            if j == EP_STEPS - 1:
                print('Episode: ', i, 'reward: %i' % (ep_r))
                plt.plot(w_track)
                plt.show()
                plt.plot(iq_track)
                plt.show()
            writer.writerow([zhuansu[time], w_track[j], iq_track[j], err_track[j]])
file.close()
file_ut.close()