# -*- coding: utf-8 -*-
"""
generate.py

Generate samples and make plots using the trained Generator.

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"


import argparse
import logging
import os

import numpy as np
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import healpy as hp
import pandas as pd

import utils
import model.truecondcgan as cgan
from  model.dataloader import FluxDataset as FD


parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="data",
                    help="Directory containing the dataset")
parser.add_argument("--test_dir", default="tests/truecondW1",
                    help="Directory containing params.json")
parser.add_argument("--restore_file", default="g_best",
                    help=("name of the file in --test_dir "
                          "containing weights to load"))


def generate(model: cgan.Generator, params: utils.Params,
             dataloader: DataLoader, norm_df: DataFrame,
             test_dir: str) -> None:
    """Use trained Generator to generate samples and make plots
    
    Args:
        model (cgan.Generator): the Generator model
        params (utils.Params): the hyperparameters used for training
        dataloader (DataLoader): the training data split up into batches
        norm_df (DataFrame): the DataFrame containing the mean and
            standard deviation used to normalize the data
        test_dir (str): directory to save plots to
    """
    model.eval()

    ri_o = []
    iz_o = []
    i_o = []

    with tqdm(total=len(dataloader)) as t:
        for i, (conditions, properties) in enumerate(dataloader):

            noise, conditions, properties = (
                Variable(torch.randn(conditions.shape[1], params.z_dim)),
                Variable(conditions[0]), Variable(properties[0]))

            if params.cuda:
                noise, conditions, properties = (
                    noise.cuda(non_blocking=True),
                    conditions.cuda(non_blocking=True),
                    properties.cuda(non_blocking=True))

            true_mag, obs_mag = properties[:, :3], properties[:, 3:]

            out = model(noise, conditions, true_mag).cpu().detach().numpy()

            out = np.array([(out[:, i] * norm_df.iloc[i + 3][1])
                            + norm_df.iloc[i + 3][0] for i in range(3)])
            out = np.swapaxes(out, 0, 1)

            ri_o.append(out[:, 0] - out[:, 1])
            iz_o.append(out[:, 1] - out[:, 2])
            i_o.append(out[:, 1])
            
            t.update()

    ri_o = np.array(ri_o).reshape(-1)
    iz_o = np.array(iz_o).reshape(-1)
    i_o = np.array(i_o).reshape(-1)

    logging.info("Making plots...")
    
    plt.rcParams.update({"font.size": 18, "figure.figsize": (10,7)})

    plt.figure()
    plt.hist2d(i_o, ri_o, bins=100, range=[[20, 25], [-2, 2]])
    plt.xlabel("$i_\mathrm{obs.}$")
    plt.ylabel("$(r-i)_\mathrm{obs.}$")
    plt.clim(0, 2500)
    plt.colorbar()
    plt.savefig(os.path.join(test_dir, "ri_i_o.png"))

    plt.figure()
    plt.hist2d(iz_o, ri_o, bins=100, range=[[-2, 2], [-2, 2]])
    plt.xlabel("$(i-z)_\mathrm{obs.}$")
    plt.ylabel("$(r-i)_\mathrm{obs.}$")
    plt.clim(0, 6000)
    plt.colorbar()
    plt.savefig(os.path.join(test_dir, "ri_iz_o.png"))

    logging.info(f"- Eval metrics: r-i_ob med={np.median(ri_o)}, "
                 f"std={np.std(ri_o)}; i-z_ob med={np.median(iz_o)}, "
                 f"std={np.std(iz_o)}")

    return None


if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join(args.test_dir, "params.json")
    assert os.path.isfile(json_path), (
        f"No json configuration file found at {json_path}")
    params = utils.Params(json_path)

    params.cuda = torch.cuda.is_available()

    torch.manual_seed(340)
    if params.cuda:
        torch.cuda.manual_seed(340)

    utils.set_logger(os.path.join(args.test_dir, "coloreval.log"))

    logging.info("Loading the dataset...")

    norm_df = pd.read_pickle(
        os.path.join(args.data_dir, "preprocessed_onlytrain/normalization.pkl")
        )

    dl = DataLoader(FD("train", params.batch_size), batch_size=1, shuffle=True,
                    num_workers=params.num_workers, pin_memory=params.cuda)

    logging.info("- done.")

    model = cgan.Generator(params).cuda() if params.cuda else (
        cgan.Generator(params))

    logging.info("Starting evaluation...")

    # Load the model's saved parameters post-training
    utils.load_checkpoint(
        os.path.join(args.test_dir, args.restore_file + ".pth.tar"), model)

    generate(model, params, dl, norm_df, args.test_dir)

    logging.info("- done.")

