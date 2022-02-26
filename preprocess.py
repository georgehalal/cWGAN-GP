# -*- coding: utf-8 -*-
"""
preprocess.py

Load, preprocess, and save the data as torch tensors to be loaded by
dataloader.py

Author: George Halal
Email: halalgeorge@gmail.com
"""


__author__ = "George Halal"
__email__ = "halalgeorge@gmail.com"


import os
import argparse
from functools import reduce
import pickle as pkl

import pandas as pd
import healpy as hp
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", default="data/original",
                    help="Directory with the original dataset")
parser.add_argument("--output_dir", default="data/preprocessed",
                    help="Where to write the new dataset")


def modify_columns(df: pd.DataFrame, skymaps: list[np.ndarray],
                   skymapnames: list[str]) -> pd.DataFrame:
    """Add and remove columns from the Pandas DataFrame
    
    Args:
        df (pd.DataFrame): the Pandas DataFrame containing the
            magnitudes
        skymaps (list[np.ndarray]): the maps containing the conditions
        skymapnames (list[str]): the names of the maps

    Returns:
        (pd.DataFrame): DataFrame with the columns needed for training
    """
    df["pixel"] = hp.ang2pix(nside = 4096,
                             theta = df["unsheared/dec"].apply(
                                lambda dec: np.deg2rad(90 - dec)),
                             phi = df["unsheared/ra"].apply(
                                lambda ra: np.deg2rad(ra)))

    df = df[["pixel", "BDF_FLUX_DERED_R", "BDF_FLUX_DERED_I",
             "BDF_FLUX_DERED_Z", "unsheared/flux_r", "unsheared/flux_i",
             "unsheared/flux_z"]]

    select = np.array(df["pixel"])

    for (skymapname, skymap) in zip(skymapnames, skymaps):
        df[skymapname] = skymap[select]

    return df


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split the Pandas DataFrame into a Training test, and validation
    sets.

    Args:
        df (pd.DataFrame): the Pandas DataFrame with all the data

    Returns:
        (tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]): tuple of
            testing, training, and validation data splits as Pandas
            DataFrames
    """
    # shuffle
    df = df.sample(frac=1).reset_index(drop=True)
    split = int(0.1 * len(df))
    test_df = df.iloc[:split]
    val_df = df.iloc[split:2 * split]
    train_df = df.iloc[2 * split:]

    return test_df, train_df, val_df


def split_and_normalize(df: pd.DataFrame, output_dir: str) -> dict:
    """Split the DataFrame and Normalize the results by the
    mean and standard deviation of the training set.

    Args:
        df (pd.DataFrame): the Pandas DataFrame with all the data.
        output_dir (str): the directory to save the mean and standard
            deviation of the training set to.
    Returns:
        (dict): dictionary of the testing, training, and validation
            DataFrames.
    """
    test_df, train_df, val_df = split(df)
    
    train_mean = np.array(train_df.mean()[1:])
    train_std = np.array(train_df.std()[1:])

    train_df[[str(col) for col in train_df.columns][1:]] = ((
        train_df[[str(col) for col in train_df.columns][1:]] - train_mean)
        / train_std)
    test_df[[str(col) for col in test_df.columns][1:]] = ((
        test_df[[str(col) for col in test_df.columns][1:]] - train_mean)
        / train_std)
    val_df[[str(col) for col in val_df.columns][1:]] = ((
        val_df[[str(col) for col in val_df.columns][1:]] - train_mean)
        / train_std)

    pd.DataFrame({"train_mean": train_mean, "train_std": train_std},
                 index=[str(col) for col in train_df.columns][1:]).to_pickle(
                 os.path.join(output_dir, "normalization.pkl"))

    return {"test": test_df, "train": train_df, "val": val_df}


def apply_selections(fp: np.ndarray, fg: np.ndarray,
                     br: np.ndarray) -> np.ndarray:
    """Select pixels for which the observation is cleanest.
    
    Args:
        fp (np.ndarray): footprint of the DES patch.
        fg (np.ndarray): pixels with foreground contamination.
        br (np.ndarray): poorly observed regions.

    Returns:
        (np.ndarray): pixels for which the observation is cleanest.
    """
    
    ind_fp = np.where(fp==1)[0]
    ind_fg = np.where(fg==0)[0]
    ind_br = np.where(br<2)[0]

    return reduce(np.intersect1d, (ind_fp, ind_fg, ind_br))


def preprocess(args) -> None:
    """Prepare the data to be fed into the neural network
    """

    conditions_dir = os.path.join(args.input_dir, "conditions")
    selections_dir = os.path.join(args.input_dir, "selections")
    
    skymapnames = os.listdir(conditions_dir)
    print("Reading sky maps...")
    skymaps = [hp.read_map(
        os.path.join(conditions_dir, skymapname)) for skymapname in skymapnames]

    print("Loading Balrog DataFrame...")
    df = pd.read_pickle(os.path.join(args.input_dir, "deep_balrog.pkl"))

    print("Modifying DataFrame columns...")
    df = modify_columns(df, skymaps, skymapnames)

    print("Splitting and normalizing the dataset...")
    split_df = split_and_normalize(df, args.output_dir)
    print("Saving preprocessed dataset...")

    for dslice in ["test", "train", "val"]:
        print(f"Saving preprocessed {dslice} data to {args.output_dir}")
        split_df[dslice].to_pickle(
            os.path.join(args.output_dir, f"{dslice}_data.pkl"))

    selections = os.listdir(selections_dir)
    print("Selecting sky area...")
    for filename in selections:
        if "footprint" in filename:
            fp = hp.read_map(os.path.join(selections_dir, filename))
        if "badregion" in filename:
            br = hp.read_map(os.path.join(selections_dir, filename))
        if "foreground" in filename:
            fg = hp.read_map(os.path.join(selections_dir, filename))

    selection_pixels = apply_selections(fp, fg, br)
    print("Saving selection pixels...")
    np.save(
        os.path.join(args.output_dir, "selection_pixels"), selection_pixels)


if __name__ == "__main__":
    args = parser.parse_args()
    
    assert os.path.isdir(args.input_dir), (
        f"Couldn't find the dataset at {args.input_dir}")
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print(f"Warning: output dir {args.output_dir} already exists")

    preprocess(args)
