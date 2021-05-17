import argparse
import os
import pandas as pd
import pickle as pkl
import healpy as hp
import numpy as np
from functools import reduce


parser = argparse.ArgumentParser()
parser.add_argument('--input_dir', default='data/original', \
                    help="Directory with the original dataset")
parser.add_argument('--output_dir', default='data/preprocessed', \
                    help="Where to write the new dataset")


def modify_columns(df, skymaps, skymapnames):
    """Add and remove columns from the Pandas DataFrame
    
    Args:
        df: the Pandas DataFrame containing the magnitudes
        skymaps: the maps containing the conditions
        skymapnames: the names of the maps
    """

    df['pixel'] = hp.ang2pix(nside = 4096, \
                             theta = df['unsheared/dec'].apply(lambda dec: \
                                                               np.deg2rad(90 - dec)), \
                             phi = df['unsheared/ra'].apply(lambda ra: np.deg2rad(ra)))
    df = df[['pixel','BDF_FLUX_DERED_R','BDF_FLUX_DERED_I','BDF_FLUX_DERED_Z',\
             'unsheared/flux_r','unsheared/flux_i','unsheared/flux_z']]
    select = np.array(df['pixel'])
    for (skymapname, skymap) in zip(skymapnames, skymaps):
        df[skymapname] = skymap[select]
    return df


def split(df):
    """Split the Pandas DataFrame into a Training
    test, and validation sets.

    Args:
        df: the Pandas DataFrame with all the data
    """

    df = df.sample(frac=1).reset_index(drop=True)
    split = int(0.1 * len(df))
    test_df = df.iloc[:split]
    val_df = df.iloc[split:2*split]
    train_df = df.iloc[2*split:]
    return test_df, train_df, val_df


def split_and_normalize(df, output_dir):
    """Split the DataFrame and Normalize the results by the
    mean and standard deviation of the training set.

    Args:
        df: the Pandas DataFrame with all the data
        output_dir: the directory to save the mean and standard
            deviation of the training set to
    """
    
    test_df, train_df, val_df = split(df)
    train_mean = np.array(train_df.mean()[1:])
    train_std = np.array(train_df.std()[1:])
    test_df[[str(col) for col in test_df.columns][1:]] = \
        (test_df[[str(col) for col in test_df.columns][1:]] - train_mean) / train_std
    train_df[[str(col) for col in train_df.columns][1:]] = \
        (train_df[[str(col) for col in train_df.columns][1:]] - train_mean) / train_std
    val_df[[str(col) for col in val_df.columns][1:]] = \
        (val_df[[str(col) for col in val_df.columns][1:]] - train_mean) / train_std
    pd.DataFrame({'train_mean': train_mean, 'train_std': train_std}, \
                 index=[str(col) for col in train_df.columns][1:]).to_pickle(\
                        os.path.join(output_dir,'normalization.pkl'))
    return {'test': test_df, 'train': train_df, 'val': val_df}


def apply_selections(fp, fg, br):
    """Select pixels for which the observation is cleanest.
    
    Args:
        fp: footprint of the DES patch
        fg: pixels with foreground contamination
        br: poorly observed regions
    """
    
    ind_fp = np.where(fp==1)[0]
    ind_fg = np.where(fg==0)[0]
    ind_br = np.where(br<2)[0]
    return reduce(np.intersect1d, (ind_fp, ind_fg, ind_br))


def preprocess(args):
    """Prepare the data to be fed into the neural network"""

    conditions_dir = os.path.join(args.input_dir, 'conditions')
    selections_dir = os.path.join(args.input_dir, 'selections')
    
    skymapnames = os.listdir(conditions_dir)
    print("Reading sky maps...")
    skymaps = [hp.read_map(os.path.join(conditions_dir,\
                                        skymapname)) for skymapname in skymapnames]
    print("Loading Balrog DataFrame...")
    df = pd.read_pickle(os.path.join(args.input_dir, 'deep_balrog.pkl'))
    print("Modifying DataFrame columns...")
    df = modify_columns(df, skymaps, skymapnames)
    print("Splitting and normalizing the dataset...")
    split_df = split_and_normalize(df, args.output_dir)
    print("Saving preprocessed dataset...")

    for dslice in ['test', 'train', 'val']:
        print("Saving preprocessed {} data to {}".format(dslice, \
                                                         args.output_dir))
        split_df[dslice].to_pickle(os.path.join(args.output_dir,\
                                                '{}_data.pkl'.format(dslice)))

    selections = os.listdir(selections_dir)
    print("Selecting sky area...")
    for filename in selections:
        if 'footprint' in filename:
            fp = hp.read_map(os.path.join(selections_dir, filename))
        if 'badregion' in filename:
            br = hp.read_map(os.path.join(selections_dir, filename))
        if 'foreground' in filename:
            fg = hp.read_map(os.path.join(selections_dir, filename))
    selection_pixels = apply_selections(fp, fg, br)
    print("Saving selection pixels...")
    np.save(os.path.join(args.output_dir,'selection_pixels'), selection_pixels)


if __name__ == '__main__':
    args = parser.parse_args()
    
    assert os.path.isdir(args.input_dir), \
        "Couldn't find the dataset at {}".format(args.input_dir)
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)
    else:
        print("Warning: output dir {} already exists".format(args.output_dir))

    preprocess(args)
