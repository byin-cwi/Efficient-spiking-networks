import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

surrograte_type = 'MG'
print('gradient type: ', surrograte_type)

gamma = 0.5
lens = 0.5
R_m = 1

beta_value = 0.184
b_j0_value = 1.6

# beta_value = 1.8
# b_j0_value = .1

# beta_value = .2#1.8
# b_j0_value = .1

def gaussian(x, mu=0., sigma=.5):
    return torch.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / torch.sqrt(2 * torch.tensor(math.pi)) / sigma

# define approximate firing function

class ActFun_adp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):  # input = membrane potential- threshold
        ctx.save_for_backward(input)
        return input.gt(0).float()  # is firing ???

    @staticmethod
    def backward(ctx, grad_output):  # approximate the gradients
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        # temp = abs(input) < lens
        scale = 6.0
        hight = .15
        if surrograte_type == 'G':
            temp = torch.exp(-(input**2)/(2*lens**2))/torch.sqrt(2*torch.tensor(math.pi))/lens
        elif surrograte_type == 'MG':
            temp = gaussian(input, mu=0., sigma=lens) * (1. + hight) \
                - gaussian(input, mu=lens, sigma=scale * lens) * hight \
                - gaussian(input, mu=-lens, sigma=scale * lens) * hight
        elif surrograte_type =='linear':
            temp = F.relu(1-input.abs())
        elif surrograte_type == 'slayer':
            temp = torch.exp(-5*input.abs())
        return grad_input * temp.float() * gamma
    
    
act_fun_adp = ActFun_adp.apply    
    
def mem_update_adp(inputs, mem, spike, tau_adp, b, tau_m, dt=1, isAdapt=1,device=None):

    alpha = torch.exp(-1. * dt / tau_m).to(device)
    ro = torch.exp(-1. * dt / tau_adp).to(device)
    if isAdapt:
        beta = beta_value
    else:
        beta = 0.

    b = ro * b + (1 - ro) * spike
    B = b_j0_value + beta * b

    mem = mem * alpha + (1 - alpha) * R_m * inputs - B * spike * dt
    inputs_ = mem - B
    # spike = F.relu(inputs_)  
    spike = act_fun_adp(inputs_)  
    return mem, spike, B, b


def output_Neuron(inputs, mem, tau_m, dt=1,device=None):
    """
    The read out neuron is leaky integrator without spike
    """
    alpha = torch.exp(-1. * dt / tau_m).to(device)
    mem = mem *alpha +  (1-alpha)*inputs
    return mem