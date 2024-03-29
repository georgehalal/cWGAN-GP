# -*- coding: utf-8 -*-
"""
train_and_evaluate.py

Train the model on the training dataset, evaluate on the validation
dataset, and save plots of the metrics across training epochs.

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"


import argparse
import logging
import os
from typing import Callable, Optional
import pickle as pkl

import numpy as np
import torch
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd
import healpy as hp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import utils
import model.cWGAN as cgan
from model.dataloader import FluxDataset as FD


plt.rcParams.update({"font.size": 15, "figure.figsize": (10, 6)})
parser = argparse.ArgumentParser()
parser.add_argument("--test_dir", default="tests/truecondW1",
                    help="Directory containing params.json")
parser.add_argument("--gen_restore_file", default=None,
                    help=("Optional, name of the file in --test_dir containing"
                          " generator weights to reload before training"))
parser.add_argument("--disc_restore_file", default=None,
                    help=("Optional, name of the file in --test_dir containing"
                          " discriminator weights to reload before training"))


def calc_gradient_penalty(params: utils.Params, d_model: cgan.Discriminator,
                          obs_mag: torch.tensor, g_out: torch.tensor,
                          conditions: torch.tensor,
                          true_mag: torch.tensor) -> torch.tensor:
    """Calculate the gradient penalty

    Args:
        params (utils.Params): the hyperparameters used for training
        d_model (cgan.Discriminator): the Discriminator model
        obs_mag (torch.tensor): the ground truth observed galaxy magnitudes
        g_out (torch.tensor): the output of the Generator at this step
        conditions (torch.tensor): the observing conditions used as inputs
        true_mag (torch.tensor): the ground truth true galaxy magnitudes

    Returns:
        (torch.tensor): gradient penalty
    """
    alpha = torch.rand(conditions.shape[0], 1)
    alpha = alpha.expand(conditions.shape[0], 3).contiguous()
    if params.cuda:
        alpha = alpha.cuda(non_blocking=True)

    interpolates = alpha * obs_mag.detach() + ((1 - alpha) * g_out.detach())
    interpolates.requires_grad_(True)

    disc_interpolates = d_model(interpolates, conditions, true_mag)
     
    ones = torch.ones(disc_interpolates.size())
    if params.cuda:
        ones = ones.cuda(non_blocking=True)

    gradients = grad(outputs=disc_interpolates, inputs=interpolates,
                     grad_outputs=ones, create_graph=True, retain_graph=True,
                     only_inputs=True)[0]

    gradients = gradients.view(gradients.size(0), -1) 
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    return gradient_penalty


def disc_train_step(params: utils.Params, d_model: cgan.Discriminator,
                    g_model: cgan.Generator, d_optimizer: optim.Adam,
                    true_mag: torch.tensor, obs_mag: torch.tensor,
                    conditions: torch.tensor) -> float:
    """One training step of the Discriminator model

    Args:
        params (utils.Params): the hyperparameters used for training
        d_model (cgan.Discriminator): the Discriminator model
        g_model (cgan.Generator): the Generator model
        d_optimizer (optim.Adam): the optimizer for the Discriminator
            model
        true_mag (torch.tensor): the ground truth true galaxy magnitudes
        obs_mag (torch.tensor): the ground truth observed galaxy
            magnitudes
        conditions (torch.tensor): the observing conditions used as
            inputs

    Returns:
        (float): Discriminator loss after one training step
    """
    d_optimizer.zero_grad()

    z = Variable(torch.randn(conditions.shape[0], params.z_dim))

    conditions.requires_grad_(True)
    true_mag.requires_grad_(True)
    obs_mag.requires_grad_(True)
    z.requires_grad_(True)

    if params.cuda:
        z = z.cuda(non_blocking=True)
    
    d_trueout = d_model(obs_mag, conditions, true_mag).squeeze()
    true_cost = d_trueout.mean()

    g_out = g_model(z, conditions, true_mag)
    d_generatedout = d_model(g_out, conditions, true_mag).squeeze()
    gen_cost = d_generatedout.mean()

    gradient_penalty = calc_gradient_penalty(params, d_model, obs_mag,
                                             g_out, conditions, true_mag)

    d_cost = gen_cost - true_cost + gradient_penalty
    d_cost.backward()
    d_optimizer.step()

    return d_cost.item()


def gen_train_step(params: utils.Params, metrics: dict,
                   d_model: cgan.Discriminator, g_model: cgan.Generator,
                   g_optimizer: optim.Adam, true_mag: torch.tensor,
                   obs_mag: torch.tensor,
                   conditions: torch.tensor) -> tuple[float, float]:
    """One training step of the Generator model

    Args:
        params (utils.Params): the hyperparameters used for training
        metrics (dict): the metrics to evaluate the model on
        d_model (cgan.Discriminator): the Discriminator model
        g_model (cgan.Generator): the Generator model
        g_optimizer (optim.Adam): the optimizer for the Generator model
        true_mag (torch.tensor): the ground truth true galaxy magnitudes
        obs_mag (torch.tensor): the ground truth observed galaxy
            magnitudes
        conditions (torch.tensor): the observing conditions used as
            inputs

    Returns:
        (tuple[float, float]): Generator loss after one training step
            and mean-square error loss
    """
    g_optimizer.zero_grad()
    z = Variable(torch.randn(conditions.shape[0], params.z_dim))
    z.requires_grad_(True)
    conditions.requires_grad_(True)
    true_mag.requires_grad_(True)

    if params.cuda:
        z = z.cuda(non_blocking=True)

    g_out = g_model(z, conditions, true_mag)
    d_out = d_model(g_out, conditions, true_mag).squeeze()
    mse_loss = metrics["MSE"](g_out, obs_mag)

    # Add a mean-squared error loss to the cost function
    g_cost = -d_out.mean() + mse_loss
    g_cost.backward()
    g_optimizer.step()

    return g_cost.item(), mse_loss.item()


def train(g_model: cgan.Generator, d_model: cgan.Discriminator,
          g_optimizer: optim.Adam, d_optimizer: optim.Adam,
          dataloader: DataLoader, metrics: dict,
          params: utils.Params) -> dict:
    """Train the Generator and Discriminator

    Args:
        g_model (cgan.Generator): the Generator model
        d_model (cgan.Discriminator): the Discriminator model
        g_optimizer (optim.Adam): the optimizer for the Generator model
        d_optimizer (optim.Adam): the optimizer for the Discriminator
            model
        dataloader (DataLoader): the training data split up into batches
        metrics (dict): the metrics to evaluate the model on
        params (utils.Params): the hyperparameters used for training

    Returns:
        (dict): a dictionary of the mean of each metric.
    """
    g_model.train()
    
    summ = []
    g_loss_avg = utils.RunningAverage()
    d_loss_avg = utils.RunningAverage()

    with tqdm(total=len(dataloader)) as t:
        for i, (conditions_batch, properties_batch) in enumerate(dataloader):
            
            conditions_batch, properties_batch = (
                Variable(conditions_batch[0]), Variable(properties_batch[0]))
            
            if params.cuda:
                conditions_batch, properties_batch = (
                    conditions_batch.cuda(non_blocking=True),
                    properties_batch.cuda(non_blocking=True))

            true_batch, obs_batch = (properties_batch[:, :3],
                                     properties_batch[:, 3:])

            # Step once through the Discriminator training process
            for p in d_model.parameters():
                p.requires_grad_(True)

            d_loss = disc_train_step(params, d_model, g_model, d_optimizer,
                                     true_batch, obs_batch, conditions_batch)

            # For every 5 Discriminator training steps, step once through the 
            # Generator training process
            if i % 5 == 0:
                for p in d_model.parameters():
                    p.requires_grad_(False)

                g_loss, mse_loss = gen_train_step(
                    params, metrics, d_model, g_model, g_optimizer, true_batch,
                    obs_batch, conditions_batch)

            if i % params.save_summary_steps == 0:
                summary_batch = {"MSE": mse_loss}
                summary_batch["g_loss"] = g_loss
                summary_batch["d_loss"] = d_loss
                summ.append(summary_batch)

            g_loss_avg.update(g_loss)
            d_loss_avg.update(d_loss)

            t.set_postfix(loss=f"{g_loss_avg():05.3f}, {d_loss_avg():05.3f}")
            t.update()

    metrics_mean = {
        metric: np.mean([x[metric] for x in summ]) for metric in summ[0]}
    metrics_string = " ; ".join(
        f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Train metrics: " + metrics_string)

    return metrics_mean


def evaluate(g_model: cgan.Generator, d_model: cgan.Discriminator,
             metrics: dict, halves: torch.tensor, val_cond: torch.tensor,
             val_true: torch.tensor, val_obs: torch.tensor,
             params: utils.Params) -> dict:
    """Evaluate the metrics to specify when to save out the weights.
    
    Args:
        g_model (cgan.Generator): the Generator model
        d_model (cgan.Discriminator): the Discriminator model
        metrics (dict): the metrics to evaluate the model on
        halves (torch.tensor): tensor filled with the scalar value 0.5
        val_cond (torch.tensor): the conditional layer for the
            validation dataset
        val_true (torch.tensor): the ground truth true magnitudes for
            the validation dataset
        val_obs (torch.tensor): the ground truth observed magnitudes for
            the validation dataset
        params (utils.Params): the hyperparameters

    Returns:
        (dict): a dictionary of the mean of each metric.
    """
    g_model.eval()

    noise = Variable(torch.randn(val_cond.shape[0], params.z_dim))

    if params.cuda:
        noise = noise.cuda(non_blocking=True)
        val_true = val_true.cuda(non_blocking=True)
        val_cond = val_cond.cuda(non_blocking=True)
    
    g_out = g_model(noise, val_cond, val_true)
    d_out = d_model(g_out, val_cond, val_true).squeeze().cpu()

    metrics_mean = {"d_MSE": metrics["MSE"](d_out, halves),
                    "d_Var": metrics["VAR"](d_out),
                    "MSE": metrics["MSE"](g_out.cpu(), val_obs),
                    "main_metric": metrics["main"](
                        d_out, halves, val_cond.shape[0])}
    metrics_string = " ; ".join(
        f"{k}: {v:05.3f}" for k, v in metrics_mean.items())
    logging.info("- Eval metrics : " + metrics_string)

    return metrics_mean


def make_plot(p: list[float], name: str, y: str, test_dir: str) -> None:
    """Plot and save metrics as a function of epochs.

    Args:
        p (list[float]): metric to plot
        name (str): name to save plot to
        y (str): y-axis label name
        test_dir (str): the directory to save the plots to
    """
    plt.figure()
    plt.plot(p)
    plt.xlabel("Epochs")
    plt.ylabel(y)
    if name=="train_MSEs" or name=="val_MSEs":
        plt.ylim(0, 5)
    plt.savefig(os.path.join(test_dir, name + ".png"))

    return None

def train_and_evaluate(g_model: cgan.Generator, d_model: cgan.Discriminator,
                       train_dataloader: DataLoader, halves: torch.tensor,
                       val_cond: torch.tensor, val_true: torch.tensor,
                       val_obs: torch.tensor, g_optimizer: optim.Adam,
                       d_optimizer: optim.Adam, metrics: dict,
                       params: utils.Params, test_dir: str,
                       gen_restore_file: Optional[str] = None,
                       disc_restore_file: Optional[str] = None) -> None:
    """Train the model, evaluate the metrics, and save some plots.

    Args:
        g_model (cgan.Generator): the Generator model
        d_model (cgan.Discriminator): the Discriminator model
        train_dataloader (DataLoader): the training data split up into
            batches
        halves (torch.tensor): tensor filled with the scalar value 0.5
        val_cond (torch.tensor): the conditional layer for the
            validation dataset
        val_true (torch.tensor): the ground truth true magnitudes for
            the validation dataset
        val_obs (torch.tensor): the ground truth observed magnitudes for
            the validation dataset
        g_optimizer (optim.Adam): the optimizer for the Generator model
        d_optimizer (optim.Adam): the optimizer for the Discriminator
            model
        metrics (dict): the metrics to evaluate the model on
        params (utils.Params): the hyperparameters used for training
        test_dir (str): the directory containing the testing parameters
        gen_restore_file (str): file containing Generator parameters to
            load and continue training
        disc_restore_file (str): file containing Discriminator
            parameters to load and continue training
    """
    if gen_restore_file is not None:
        gen_restore_path = os.path.join(
            test_dir, gen_restore_file + ".pth.tar")
        logging.info(f"Restoring generator parameters from {gen_restore_path}")
        utils.load_checkpoint(gen_restore_path, g_model, g_optimizer)

    if disc_restore_file is not None:
        disc_restore_path = os.path.join(
            test_dir, disc_restore_file + ".pth.tar")
        logging.info(
            f"Restoring discriminator parameters from {disc_restore_path}")
        utils.load_checkpoint(disc_restore_path, d_model, d_optimizer)

    best_val = 9999999999.9
    d_MSEs = []
    d_Vars = []
    val_MSEs = []
    main_metrics = []
    train_MSEs = []
    g_losses = []
    d_losses = []

    for epoch in range(params.num_epochs):
        logging.info(f"Epoch {epoch + 1} / {params.num_epochs}")

        train_metrics = train(g_model, d_model, g_optimizer, d_optimizer,
                              train_dataloader, metrics, params)

        val_metrics = evaluate(g_model, d_model, metrics, halves, val_cond,
                               val_true, val_obs, params)

        train_MSEs.append(train_metrics["MSE"].item())
        g_losses.append(train_metrics["g_loss"].item())
        d_losses.append(train_metrics["d_loss"].item())
        d_MSEs.append(val_metrics["d_MSE"].item())
        d_Vars.append(val_metrics["d_Var"].item())
        val_MSEs.append(val_metrics["MSE"].item())
        main_metrics.append(val_metrics["main_metric"].item())

        if epoch > 50:
            val_main = val_metrics["main_metric"]
            is_best = val_main <= best_val

            utils.save_checkpoint(
                {"epoch": epoch + 1, "state_dict": g_model.state_dict(),
                 "optim_dict": g_optimizer.state_dict()},
                is_best=is_best, checkpoint=test_dir, model="gen")
            utils.save_checkpoint(
                {"epoch": epoch + 1, "state_dict": d_model.state_dict(),
                 "optim_dict": d_optimizer.state_dict()},
                is_best=is_best, checkpoint=test_dir, model="disc")

            if is_best:
                logging.info("- Found new best validation metric")
                best_val = val_main

                best_json_path = os.path.join(
                    test_dir, "metrics_val_best.json")
                utils.save_dict_to_json(val_metrics, best_json_path)

        last_json_path = os.path.join(test_dir, "metrics_val_last.json")
        utils.save_dict_to_json(val_metrics, last_json_path)

    make_plot(train_MSEs, "train_MSEs", "Train MSE(gen. out, truth)", test_dir)
    make_plot(g_losses, "g_losses", "Generator Loss", test_dir)
    make_plot(d_losses, "d_losses", "Discriminator Loss", test_dir)
    make_plot(d_MSEs, "d_MSEs", "MSE(disc. out, 0.5)", test_dir)
    make_plot(d_Vars, "d_Vars", "Var(disc. out)", test_dir)
    make_plot(val_MSEs, "val_MSEs", "Val MSE(gen. out, truth)", test_dir)
    make_plot(main_metrics, "main_metrics",
              "N*MSE(disc. out, 0.5)/(N+1) + Var(disc. out)/(N+1)", test_dir)

    return None


if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join(args.test_dir, "params.json")
    assert os.path.isfile(json_path), (
        f"No json configuration file found at {json_path}")
    params = utils.Params(json_path)

    # Check whether a GPU is available
    params.cuda = torch.cuda.is_available()

    torch.manual_seed(340)

    if params.cuda:
        torch.cuda.manual_seed(340)

    utils.set_logger(os.path.join(args.test_dir, "train.log"))

    logging.info("Loading the datasets...")

    # The batch size is set to 1 here because the dataloader already
    # splits the data up into batches
    train_dl = DataLoader(
        FD("train", params.batch_size), batch_size=1, shuffle=True,
        num_workers=params.num_workers, pin_memory=params.cuda)
    val_dl = DataLoader(
        FD("val", 10000), batch_size=1, shuffle=False,
        num_workers=params.num_workers, pin_memory=params.cuda) 
    val_cond, val_out = next(iter(val_dl))
    val_out, val_cond = Variable(val_out[0]), Variable(val_cond[0])
    val_true, val_obs = (val_out[:, :3].contiguous(),
                         val_out[:, 3:].contiguous())
    del val_out, val_dl
    halves = Variable(torch.ones(val_cond.shape[0]) * 0.5)

    logging.info("- done.")

    # Load the Generator and Discriminator
    g_model = cgan.Generator(params).cuda() if params.cuda else (
        cgan.Generator(params))
    d_model = cgan.Discriminator(params).cuda() if params.cuda else (
        cgan.Discriminator(params))
    
    logging.info(g_model)
    logging.info(d_model)
    g_optimizer = optim.Adam(g_model.parameters(), lr=params.learning_rate,
                             betas=(params.beta1, 0.999))
    d_optimizer = optim.Adam(d_model.parameters(), lr=params.learning_rate,
                             betas=(params.beta1, 0.999))

    metrics = cgan.metrics

    logging.info(f"Starting training for {params.num_epochs} epoch(s)")
    train_and_evaluate(g_model, d_model, train_dl, halves, val_cond, val_true,
                       val_obs, g_optimizer, d_optimizer, metrics, params,
                       args.test_dir, args.gen_restore_file,
                       args.disc_restore_file)

