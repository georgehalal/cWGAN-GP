# -*- coding: utf-8 -*-
"""
cWGAN.py

Contains the generator and discriminator models and the metrics used
to evaluate them.

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"


import sys
sys.path.append("../")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from utils import Params


class Generator(nn.Module):
    def __init__(self, params: Params) -> None:
        """Define the building blocks of the Generator

        Args:
            params (Params): hyperparameters
        """
        super(Generator, self).__init__()

        if params.gen_act_func == "relu":
            self.func_params = (True)
            self.act_func = nn.ReLU

        elif params.gen_act_func == "leaky_relu":
            self.func_params = (0.2, True)
            self.act_func = nn.LeakyReLU

        elif params.gen_act_func == "elu":
            self.func_params = (True)
            self.act_func = nn.ELU

        self.noise = nn.Sequential(
            nn.Linear(params.z_dim, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(*self.func_params))

        self.cond = nn.Sequential(
            nn.Linear(6, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(*self.func_params))

        self.true = nn.Sequential(
            nn.Linear(3, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(*self.func_params))

        self.out = nn.Sequential(
            nn.Linear(params.num_nodes * 3 // 2, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, 3))
    
        return None

    def forward(self, z: torch.tensor, y: torch.tensor,
                t: torch.tensor) -> torch.tensor:
        """Define how the Generator operates on the input batch
        
        Args:
            z (torch.tensor): Noise layer
            y (torch.tensor): Conditional layer
            t (torch.tensor): Ground truth magnitudes layer

        Returns:
            (torch.tensor): observed galaxy magnitudes
        """
        z = self.noise(z)
        y = self.cond(y)
        t = self.true(t)
        x = self.out(torch.cat([z,y,t],-1))

        return x


class Discriminator(nn.Module):
    def __init__(self, params: Params):
        """Define the building blocks of the Discriminator

        Args:
            params (Params): hyperparameters
        """
        super(Discriminator, self).__init__()

        if params.gen_act_func == "relu":
            self.func_params = (True)
            self.act_func = nn.ReLU

        elif params.gen_act_func == "leaky_relu":
            self.func_params = (0.2, True)
            self.act_func = nn.LeakyReLU

        elif params.gen_act_func == "elu":
            self.func_params = (True)
            self.act_func = nn.ELU
        
        self.genout = nn.Sequential(
            nn.Linear(3, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(*self.func_params))
        
        self.cond = nn.Sequential(
            nn.Linear(6, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(*self.func_params))

        self.true = nn.Sequential(
            nn.Linear(3, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes // 2),
            self.act_func(*self.func_params))

        self.out = nn.Sequential(
            nn.Linear(params.num_nodes * 3 // 2, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, params.num_nodes),
            self.act_func(*self.func_params),
            nn.Linear(params.num_nodes, 1))

        return None
      
    def forward(self, x: torch.tensor, y: torch.tensor,
                t: torch.tensor) -> torch.tensor:
        """Define how the Discriminator operates on the input batch
        
        Args:
            x (torch.tensor): Output of the Generator
            y (torch.tensor): Conditional layer
            t (torch.tensor): Ground truth magnitudes

        Returns:
            (torch.tensor): real or generated
        """
        x = self.genout(x)
        y = self.cond(y)
        t = self.true(t)
        out = self.out(torch.cat([x, y, t], -1))

        return out


def MSE(out: torch.tensor, truth: torch.tensor) -> torch.tensor:
    """Mean-squared error loss

    Args:
        out (torch.tensor): output of the network
        truth (torch.tensor): ground truth to compare it to

    Returns:
        (torch.tensor): Mean-squared error loss
    """
    mse = nn.MSELoss()
    return mse(out, truth)


def VAR(out: torch.tensor) -> torch.tensor:
    """Calculate the variance of the output

    Args:
        out (torch.tensor): output of the network

    Returns:
        (torch.tensor): variance of the output
    """
    return torch.var(out)


def main_metric(out: torch.tensor, truth: torch.tensor,
                N: int) -> torch.tensor:
    """Define a metric to be calculated and saved over the epochs

    Args:
        out (torch.tensor): output of the network
        truth (torch.tensor): ground truth to compare it to
        N (int): number of samples

    Returns:
        (torch.tensor): the main metric
    """
    return N * MSE(out, truth) / (N+1) + VAR(out) / (N+1)


metrics: dict = {
    "MSE": MSE,
    "VAR": VAR,
    "main": main_metric
}   


