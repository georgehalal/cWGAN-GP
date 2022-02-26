# -*- coding: utf-8 -*-
"""
utils.py

Contains useful functions and classes for the other scripts
in this project.
"""


import json
import logging
import os
import shutil
from typing import Optional

import torch
       
    
def set_logger(log_path: str) -> None:
    """Log output in the terminal to a file
    
    Args:
        log_path (str): where to save the log file
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s:%(levelname)s: %(message)s"))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(stream_handler)


def save_dict_to_json(d: dict, json_path: str) -> None:
    """Save dictionary of floats to a json file

    Args:
        d (dict): dictionary of floats
        json_path (str): where to save the json file
    """
    with open(json_path, "w") as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state: dict, is_best: bool, checkpoint: str,
                    model: Optional[str] = None) -> None:
    """Save the model and training parameters

    Args:
        state (dict): model's state dictionary
        is_best (bool): whether the parameters correspond to the best
            model so far
        checkpoint (str): where to save the parameters
        model (Optional[str]): string specifying whether the model is
            a generator or a discriminator
    """
    if model == "gen":
        last = "g_last.pth.tar"
        best = "g_best.pth.tar"
    elif model == "disc":
        last = "d_last.pth.tar"
        best = "d_best.pth.tar"
    else:
        last = "last.pth.tar"
        best = "best.pth.tar"
    filepath = os.path.join(checkpoint, last)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! "
              f"Making directory {checkpoint}")
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, best))


def load_checkpoint(checkpoint: str, model: torch.nn.Module,
                    optimizer: Optional[torch.optim.Optimizer] = None):
    """Load model parameters (and, if provided, optimizer dict)

    Args:
        checkpoint (str): filename to load
        model (torch.nn.Module): the model corresponding to the
            parameters loaded
        optimizer (Optional[torch.optim.Optimizer]): optimizer at
            checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint["state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optim_dict"])

    return checkpoint


class Params():
    """Load hyperparameters from a json file in a dict-like structure
    """

    def __init__(self, json_path: str):
        """Initialize the class by loading the hyperparameters from the
        json file.

        Args:
            json_path (str): the path and name of the json file to load
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path: str):
        """Save the hyperparameters to the json file

        Args:
            json_path (str): the path and name of the json file to save
                the hyperparameters in.
        """
        with open(json_path, "w") as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path: str):
        """Update the dictionary with the hyperparameters loaded from
        the json file.

        Args:
            json_path (str): the path and name of the json file to load
        """
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class RunningAverage():
    """Calculate the running average
    """
    
    def __init__(self) -> None:
        """Initialize the number of steps and total to 0
        """
        self.steps = 0
        self.total = 0
    
    def update(self, val: float) -> None:
        """Update the total and number of steps.
        
        Args:
            val (float): value to use for updating the average.
        """
        self.total += val
        self.steps += 1
    
    def __call__(self) -> float:
        """Return the value of the running average when called.

        Returns:
            (float): the value of the running average
        """
        return self.total / float(self.steps)
 
