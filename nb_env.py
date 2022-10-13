#%%
import gym
from gym import spaces 
import numpy as np
import torch
use_cuda = torch.cuda.is_available()
device   = torch.device("cuda" if use_cuda else "cpu")
#%%
class PmsmEnv(gym.Env):
    def __init__(self,f,g,C_w,C_u,w_t_hat,dt=0.01,iq=None,w=None):
        self.f = f
        self.g = g
        self.C_w = C_w
        self.C_u = C_u
        self.dt = dt
        self.iq = iq
        self.w = w
        self.w_t_hat = w_t_hat
        if self.iq is not None and self.w is not None:
            self.x = torch.stack([w,iq]).unsqueeze(0)
        self.action_space = spaces.Box(0,1,shape=(2,1),dtype=np.float32)

    def reset(self, seed=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.w = torch.rand(1).uniform_(0,2800)
        self.iq = torch.rand(1).uniform_(0,12)
        self.x = torch.stack([self.w,self.iq])
        return self.w, self.iq

    def step(self,theta,tolerance=0): 
        ut= theta.detach()*(self.w_t_hat-self.w)
        reward = -self.C_w*torch.square((self.w-self.w_t_hat))-self.C_u*torch.square(ut)
        next_state = self.x.T + (self.f@self.x.T+self.g*ut)*self.dt
        self.w = torch.clamp(next_state[0],0,2800).to(device)
        self.iq= torch.clamp(next_state[1],0,12).to(device)
        next_state = torch.stack([self.w,self.iq]).reshape(2,1)
        self.x = next_state.T
        return reward.unsqueeze(0), next_state, False, ut
    
    
#%%

#%%
