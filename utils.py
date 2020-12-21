import json
import logging
import os
import shutil
import torch
       
    
def set_logger(log_path):
    """Log output in the terminal to a file
    
    Args:
        log_path: where to save the log file
    """
    
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Save dictionary of floats to a json file

    Args:
        d: dictionary of floats
        json_path: where to save the json file
    """

    with open(json_path, 'w') as f:
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint, model=None):
    """Save the model and training parameters

    Args:
        state: model's state dictionary
        is_best: whether the parameters correspond to the best model so far
        checkpoint: where to save the parameters
    """

    if model == 'gen':
        last = 'g_last.pth.tar'
        best = 'g_best.pth.tar'
    elif model == 'disc':
        last = 'd_last.pth.tar'
        best = 'd_best.pth.tar'
    else:
        last = 'last.pth.tar'
        best = 'best.pth.tar'
    filepath = os.path.join(checkpoint, last)
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, best))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Load model parameters (and optimizer dict if provided)

    Args:
        checkpoint: filename to load
        model: the model corresponding to the parameters loaded
        optimizer: (optional) optimizer at checkpoint
    """

    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint


class Params():
    # Load hyperparameters from a json file in a dict-like structure

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        return self.__dict__


class RunningAverage():
    # Calculate the running average
    
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
 
