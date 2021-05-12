import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F
from SRNN_layers.spike_neuron import mem_update_adp

b_j0 = 1.6

class spike_cov2D(nn.Module):
    def __init__(self,
                 input_size,output_dim, kernel_size=5,strides=1,
                 pooling_type = None,pool_size = 2, pool_strides =2,
                 tauM = 20,tauAdp_inital =100, tau_initializer = 'normal',tauM_inital_std = 5,tauAdp_inital_std = 5,
                 is_adaptive=1,device='cpu'):
        
        super(spike_cov2D, self).__init__()
        # input_size = [c,w,h]
        self.input_size = input_size
        self.input_dim = input_size[0]
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.device = device
        
        if pooling_type is not None: 
            if pooling_type =='max':
                self.pooling = nn.MaxPool2d(kernel_size=pool_size, stride=pool_strides, padding=1)
            elif pooling_type =='avg':
                self.pooling = nn.AvgPool2d(kernel_size=pool_size, stride=pool_strides, padding=1)
        else:
            self.pooling = None
        
        self.conv= nn.Conv2d(input_dim,output_dim,kernel_size=kernel_size,stride=strides)
        
        self.output_size = self.compute_output_size()
        
        self.tau_m = nn.Parameter(torch.Tensor(self.output_size))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_size))
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
            nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
    
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.tau_m,self.tau_adp]
    
    def set_neuron_state(self,batch_size):
        self.mem = torch.rand(batch_size,self.output_size).to(self.device)
        self.spike = torch.zeros(batch_size,self.output_size).to(self.device)
        self.b = (torch.ones(batch_size,self.output_size)*b_j0).to(self.device)
    
    def forward(self,input_spike):
        d_input = self.conv(input_spike.float())
        if self.pooling is not None: 
            d_input = self.pool(d_input)
        self.mem,self.spike,theta,self.b = mem_update_adp(d_input,self.mem,self.spike,self.tau_adp,self.b,self.tau_m,device=self.device,isAdapt=self.is_adaptive)
        
        return self.mem,self.spike
    
    def compute_output_size(self):
        x_emp = torch.randn([1,self.input_size[0],self.input_size[1],self.input_size[2]])   
        out = self.conv(x_emp)
        if self.pooling is not None: out=self.pooling(out)
        # print(self.name+'\'s size: ', out.shape[1:])
        return out.shape[1:]

class spike_cov1D(nn.Module):
    def __init__(self,
                 input_size,output_dim, kernel_size=5,strides=1,
                 pooling_type = None,pool_size = 2, pool_strides =2,dilation=1,
                 tauM = 20,tauAdp_inital =100, tau_initializer = 'normal',tauM_inital_std = 5,tauAdp_inital_std = 5,
                 is_adaptive=1,device='cpu'):
        
        super(spike_cov1D, self).__init__()
        # input_size = [c,h]
        self.input_size = input_size
        self.input_dim = input_size[0]
        self.output_dim = output_dim
        self.is_adaptive = is_adaptive
        self.dilation = dilation
        self.device = device
        
        if pooling_type is not None: 
            if pooling_type =='max':
                self.pooling = nn.MaxPool1d(kernel_size=pool_size, stride=pool_strides, padding=1)
            elif pooling_type =='avg':
                self.pooling = nn.AvgPool1d(kernel_size=pool_size, stride=pool_strides, padding=1)
        else:
            self.pooling = None
        
        self.conv= nn.Conv1d(self.input_dim,self.output_dim,kernel_size=kernel_size,stride=strides,
                            padding=(np.ceil(((kernel_size-1)*self.dilation)/2).astype(int),),
                            dilation=(self.dilation,))
        
        self.output_size = self.compute_output_size()
        
        self.tau_m = nn.Parameter(torch.Tensor(self.output_size))
        self.tau_adp = nn.Parameter(torch.Tensor(self.output_size))
        
        if tau_initializer == 'normal':
            nn.init.normal_(self.tau_m,tauM,tauM_inital_std)
            nn.init.normal_(self.tau_adp,tauAdp_inital,tauAdp_inital_std)
    
    def parameters(self):
        return [self.dense.weight,self.dense.bias,self.tau_m,self.tau_adp]
    
    def set_neuron_state(self,batch_size):
        self.mem = (torch.zeros(batch_size,self.output_size[0],self.output_size[1])*b_j0).to(self.device)
        self.spike = torch.zeros(batch_size,self.output_size[0],self.output_size[1]).to(self.device)
        self.b = (torch.ones(batch_size,self.output_size[0],self.output_size[1])*b_j0).to(self.device)
    
    def forward(self,input_spike):
        d_input = self.conv(input_spike.float())
        if self.pooling is not None: 
            d_input = self.pooling(d_input)
        
        self.mem,self.spike,theta,self.b = mem_update_adp(d_input,self.mem,self.spike,self.tau_adp,self.b,self.tau_m,device=self.device,isAdapt=self.is_adaptive)
        
        return self.mem,self.spike
    
    def compute_output_size(self):
        x_emp = torch.randn([1,self.input_size[0],self.input_size[1]])   
        out = self.conv(x_emp)
        if self.pooling is not None: out=self.pooling(out)
        # print(self.name+'\'s size: ', out.shape[1:])
        return out.shape[1:]