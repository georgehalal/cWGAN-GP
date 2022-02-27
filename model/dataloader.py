# -*- coding: utf-8 -*-
"""
dataloader.py

A class for loading a sample of the sky conditions and observed galaxy
magnitudes.

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"


import os
import pdb
import pickle as pkl

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


class FluxDataset(Dataset):
    def __init__(self, dslice: str, batchsize: int = 100) -> None:
        """Each 100 data samples are saved in a different torch file
        
        Args:
            dslice (str): data slice (train, test, or val)
            batchsize (int): batch size
        """
        super().__init__()

        self.dir = "data/MagData/"
        self.dslice = dslice
        self.mult = batchsize // 100 # multiple of 100
        self.valsize = 10000 // batchsize

        return None

    def __len__(self) -> int:
        """Return the size of the dataset

        Returns:
            (int): size of the dataset
        """
        if self.dslice == "val":
            return self.valsize
        elif self.dslice == "train":
            return 1704537 // (self.mult * 100) - self.valsize

    def __getitem__(self, idx: int) -> tuple[torch.tensor, torch.tensor]:
        """Load and concatenate batches of 100 samples
        
        Args:
            idx (int): index

        Returns:
            (tuple[torch.tensor, torch.tensor]): sky conditions and
                observed galaxy magnitudes
        """
        if self.dslice == "train":
            idx += self.valsize

        filenum = idx * self.mult
        Dir = f"Data{filenum // 100:03d}/"
        condname = f"cond/cond{filenum:05d}"
        outname = f"out/out{filenum:05d}"

        cond = torch.load(self.dir + Dir + condname)
        out = torch.load(self.dir + Dir + outname)
            
        if self.mult>1:            
            for i in range(self.mult-1):
                filenum += 1
                Dir = f"Data{filenum // 100:03d}/"
                condname = f"cond/cond{filenum:05d}"
                outname = f"out/out{filenum:05d}"
                cond = torch.cat(
                    (cond, torch.load(self.dir + Dir + condname)), 0)
                out = torch.cat(
                    (out, torch.load(self.dir + Dir + outname)), 0)

        return cond, out

