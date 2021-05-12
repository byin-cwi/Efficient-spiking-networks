import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from SRNN_layers.spike_neuron import *#mem_update_adp
from SRNN_layers.spike_dense import *

b_j0 = b_j0_value
class spike_rnn(nn.Module):
    def __init__(self,input_dim,output_dim,
                 tauM = 20,tauAdp_inital =100, tau_initializer = 'normal',tauM_inital_std = 5,tauAdp_inital_std = 5,
                 is_adaptive=1,device='cpu',bias=True):
        super(spike_rnn, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device

        self.b_j0 = b_j0
        self.dense = nn.Linear(input_dim,output_dim,bias=bias)
        self.recurrent = nn.Linear(output_dim,output_dim,bias=bias)
        self.tau_m = nn.Parameter(torch.Tensor(self.output_dim))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_dim))
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
            nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
        elif tau_initializer == 'multi_normal':
            self.tau_m = multi_normal_initilization(self.tau_m,tauM,tauM_inital_std)
            self.tau_adp = multi_normal_initilization(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
    
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.recurrent.weight,self.recurrent.bias,self.tau_m,self.tau_adp]
    
    def set_neuron_state(self,batch_size):
        # self.mem = (torch.rand(batch_size,self.output_dim)*self.b_j0).to(self.device)
        self.mem = Variable(torch.zeros(batch_size,self.output_dim)*self.b_j0).to(self.device)
        self.spike = Variable(torch.zeros(batch_size,self.output_dim)).to(self.device)
        self.b = Variable(torch.ones(batch_size,self.output_dim)*self.b_j0).to(self.device)
    
    def forward(self,input_spike):
        d_input = self.dense(input_spike.float()) + self.recurrent(self.spike)
        self.mem,self.spike,theta,self.b = mem_update_adp(d_input,self.mem,self.spike,self.tau_adp,self.b,self.tau_m,device=self.device,isAdapt=self.is_adaptive)
        
        return self.mem,self.spike