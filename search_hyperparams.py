import argparse
import os
from subprocess import check_call
import sys

import utils


parser = argparse.ArgumentParser()
parser.add_argument('--parent_dir', default='tests/learning_rate', help='Directory containing params.json')
parser.add_argument('--data_dir', default='data', help="Directory containing the dataset")


def make_sub_file(name, test_dir, data_dir):
    """Creat a .sh file to submit as a job
    
    Args:
        name: name of the job to run
        test_dir: directory to run the job in
        data_dir: directory containing the dataset
    """

    slurm_log = name+'.log'
    slurm_out = name+'.out'
    sbatch_script = name+'.sh'

    sbatch_file = open(os.path.join(test_dir,sbatch_script), 'w')
    sbatch_file.write("#!/bin/bash \n")
    sbatch_file.write("#SBATCH --job-name=%s \n"%name)
    sbatch_file.write("#SBATCH --workdir=%s \n"%test_dir)
    sbatch_file.write("#SBATCH --time=2-00:00:00 \n")
    sbatch_file.write("#SBATCH -p gpu \n")
    sbatch_file.write("#SBATCH --gpus=2 \n")
    sbatch_file.write("#SBATCH --mem=32GB \n")
    sbatch_file.write("#SBATCH --output=%s \n"%slurm_out)
    sbatch_file.write("\n")
    sbatch_file.write("python3 train.py --test_dir {test_dir} --data_dir {data_dir} >& {slurm_log} \n".format(test_dir=test_dir, data_dir=data_dir, slurm_log=slurm_log) )
    sbatch_file.write("\n")
    sbatch_file.close()


def submit(name, test_dir):
    """Run command to submit job
    
    Args:
        name: name of the job to run
        test_dir: directory to run the job in
    """

    sbatch_script = name+'.sh'
    os.system('sbatch %s' % os.path.join(test_dir,sbatch_script))


if __name__ == "__main__":
    args = parser.parse_args()
    json_path = os.path.join(args.parent_dir, 'params.json')
    assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
    params = utils.Params(json_path)

    # Hyperparameters to try
    gen_act_funcs = ['elu', 'leaky_relu', 'relu']
    disc_act_funcs = ['elu', 'leaky_relu', 'leaky_relu']
    learning_rates = [1e-4, 1e-3, 1e-2]
    nodes_per_layer = [20, 40, 50, 100]
    dropout_rates = [0.2, 0.3, 0.4]
    batch_sizes = [1024, 2048, 4096]
    epochs = [20, 50, 100]

    for (gen_act_func, disc_act_func) in zip(gen_act_funcs, disc_act_funcs):
        params.gen_act_func = gen_act_func
        params.disc_act_func = disc_act_func

        name = "gen_{}_disc_{}".format(gen_act_func, disc_act_func)
        test_dir = os.path.join(args.parent_dir, name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        json_path = os.path.join(test_dir, 'params.json')
        params.save(json_path)

        make_sub_file(name, test_dir, args.data_dir)
        submit(name, test_dir)

    """
    for learning_rate in learning_rates:
        params.learning_rate = learning_rate

        name = "lr_{}".format(learning_rate)
        test_dir = os.path.join(args.parent_dir, name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        json_path = os.path.join(test_dir, 'params.json')
        params.save(json_path)

        make_sub_file(name, test_dir, args.data_dir)
        submit(name, test_dir)

    for num_nodes in nodes_per_layer:
        params.num_nodes = num_nodes

        name = "nodes_{}".format(num_nodes)
        test_dir = os.path.join(args.parent_dir, name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        json_path = os.path.join(test_dir, 'params.json')
        params.save(json_path)

        make_sub_file(name, test_dir, args.data_dir)
        submit(name, test_dir)

    for dropout_rate in dropout_rates:
        params.dropout_rate = dropout_rate

        name = "dr_{}".format(dropout_rate)
        test_dir = os.path.join(args.parent_dir, name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        json_path = os.path.join(test_dir, 'params.json')
        params.save(json_path)

        make_sub_file(name, test_dir, args.data_dir)
        submit(name, test_dir)

    for batch_size in batch_sizes:
        params.batch_size = batch_size

        name = "batches_{}".format(batch_size)
        test_dir = os.path.join(args.parent_dir, name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        json_path = os.path.join(test_dir, 'params.json')
        params.save(json_path)

        make_sub_file(name, test_dir, args.data_dir)
        submit(name, test_dir)

    for num_epochs in epochs:
        params.num_epochs = num_epochs

        name = "epochs_{}".format(num_epochs)
        test_dir = os.path.join(args.parent_dir, name)
        if not os.path.exists(test_dir):
            os.makedirs(test_dir)

        json_path = os.path.join(test_dir, 'params.json')
        params.save(json_path)

        make_sub_file(name, test_dir, args.data_dir)
        submit(name, test_dir)
    """


