import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, params):
        """Define the building blocks of the Generator

        Args:
            params: hyperparameters
        """
        
        super(Generator, self).__init__()
        if params.gen_act_func == 'relu':
            self.func_params = (True)
            self.act_func = nn.ReLU
        elif params.gen_act_func == 'leaky_relu':
            self.func_params = (0.2, True)
            self.act_func = nn.LeakyReLU
        elif params.gen_act_func == 'elu':
            self.func_params = (True)
            self.act_func = nn.ELU

        self.noise = nn.Sequential(nn.Linear(params.z_dim, params.num_nodes), \
                                   self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes//2), \
                                   self.act_func(*self.func_params))

        self.cond = nn.Sequential(nn.Linear(6, params.num_nodes), \
                                  self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes//2), \
                                  self.act_func(*self.func_params))

        self.true = nn.Sequential(nn.Linear(3, params.num_nodes), \
                                  self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes//2), \
                                  self.act_func(*self.func_params))

        self.out = nn.Sequential(nn.Linear(params.num_nodes*3//2, \
                                           params.num_nodes), \
                                 self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes), \
                                 self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, 3))
    
    def forward(self, z, y, t):
        """Define how the Generator operates on the input batch
        
        Args:
            z: Noise layer
            y: Conditional layer
            t: Ground truth magnitudes layer
        """

        z = self.noise(z)
        y = self.cond(y)
        t = self.true(t)
        x = self.out(torch.cat([z,y,t],-1))
        return x


class Discriminator(nn.Module):
    def __init__(self, params):
        """Define the building blocks of the Discriminator

        Args:
            params: hyperparameters
        """

        super(Discriminator, self).__init__()
        if params.gen_act_func == 'relu':
            self.func_params = (True)
            self.act_func = nn.ReLU
        elif params.gen_act_func == 'leaky_relu':
            self.func_params = (0.2, True)
            self.act_func = nn.LeakyReLU
        elif params.gen_act_func == 'elu':
            self.func_params = (True)
            self.act_func = nn.ELU
        
        self.genout = nn.Sequential(nn.Linear(3, params.num_nodes), \
                                    self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes//2), \
                                    self.act_func(*self.func_params))
        
        self.cond = nn.Sequential(nn.Linear(6, params.num_nodes), \
                                  self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes//2), \
                                  self.act_func(*self.func_params))

        self.true = nn.Sequential(nn.Linear(3, params.num_nodes), \
                                  self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes//2), \
                                  self.act_func(*self.func_params))

        self.out = nn.Sequential(nn.Linear(params.num_nodes*3//2, params.num_nodes), \
                                 self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes), \
                                 self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, 1))
      
    def forward(self, x, y, t):
        """Define how the Discriminator operates on the input batch
        
        Args:
            x: Output of the Generator
            y: Conditional layer
            t: Ground truth magnitudes
        """

        x = self.genout(x)
        y = self.cond(y)
        t = self.true(t)
        out = self.out(torch.cat([x,y,t],-1))
        return out


def MSE(out, truth):
    """Mean-squared error loss

    Args:
        out: output of the network
        truth: ground truth to compare it to
    """
    mse = nn.MSELoss()
    return mse(out, truth)


def VAR(out):
    """Variance of the output

    Args:
        out: output of the network
    """
    return torch.var(out)


def main_metric(out, truth, N):
    """Define a metric to be calculated and saved over the epochs

    Args:
        out: output of the network
        truth: ground truth to compare it to
        N: number of samples
    """
    return N*MSE(out, truth)/(N+1) + VAR(out)/(N+1)


metrics = {
    'MSE': MSE,
    'VAR': VAR,
    'main': main_metric
}   


