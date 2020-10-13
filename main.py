import argparse
import json
import os
import sys

from global_settings import method_args, task_args

from tasks import regression
from tasks import al_selection, al_prediction
from tasks import al_oracle_pretrain, al_oracle_selection, al_oracle_prediction
from tasks.toy import compute_metrics

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Predictive Correlation Benchmarks')

    # The arguments for training.
    training_args = parser.add_argument_group('training')
    training_args.add_argument('--dataset', type=str, default='boston')
    training_args.add_argument('--method', type=str)

    training_args.add_argument('-nb', '--n_base', type=int, default=1000)
    training_args.add_argument('-bs', '--batch_size', type=int, default=5000)
    training_args.add_argument('-lr', '--learning_rate', type=float, default=0.003)
    training_args.add_argument('-e', '--epochs', type=int, default=5000)
    training_args.add_argument('-test_r', '--test_ratio', type=float, default=0.2)
    training_args.add_argument('-train_r', '--train_ratio', type=float, default=0.2)
    training_args.add_argument('-seed', '--init_seed', type=int, default=1)
    training_args.add_argument('-n_runs', '--n_runs', type=int, default=10)
    training_args.add_argument('-Lae', '--lr_ann_epochs', type=int, default=1)
    training_args.add_argument('-Lar', '--lr_ann_ratio', type=float, default=1.)

    # The arguments for logging.
    logger_args = parser.add_argument_group('logging')
    logger_args.add_argument('--print_interval', type=int, default=100)
    logger_args.add_argument('--test_interval', type=int, default=100)
    logger_args.add_argument('--expid', type=str, default='')
    logger_args.add_argument('--overwrite', action='store_true')

    # Some general arguments.
    parser.add_argument('--task', type=str, default='regression')
    parser.add_argument('--note', type=str, default='')

    # Loading the specific arguments for the corresponding method and task.
    method_args.load_args(parser, sys.argv)
    task_args.load_args(parser, sys.argv)

    args = parser.parse_args()
    if args.method == 'emhmc':
        args.epochs = args.hmc_samples * args.num_steps_between_samples + args.hmc_burnin - 1
    if args.dataset in ['kin8nm', 'naval', 'power_plant']:
        if args.method == 'gp':
            args.method = 'svgp'
        if hasattr(args, 'base_method'):
            if args.base_method == 'gp':
                args.base_method = 'svgp'

    # Construct Result Directory
    if args.expid == "":
        print("WARNING: this experiment is not being saved.")
        setattr(args, 'save', False)
    else:
        result_dir = '{}/{}/{}'.format(args.result_dir, args.task, args.expid)
        setattr(args, 'save', True)
        setattr(args, 'result_dir', result_dir)
        try:
            os.makedirs(result_dir)
        except FileExistsError:
            # val = ""

            # while val not in ['yes', 'no']:
            #     val = input(
            #         "Experiment '{}' with expid '{}' exists.  Overwrite (yes/no)? ".format(args.experiment, args.expid))
            if not args.overwrite:
                quit()

    # Save Args
    if args.save:
        with open(args.result_dir + '/args.json', 'w') as f:
            json.dump(args.__dict__, f, sort_keys=True, indent=4)

    # Run Experiment
    if args.task == 'regression':
        regression.run(args)
    elif args.task == 'al_selection':
        al_selection.run(args)
    elif args.task == 'al_prediction':
        al_prediction.run(args)
    elif args.task == 'al_oracle_pretrain':
        al_oracle_pretrain.run(args)
    elif args.task == 'al_oracle_selection':
        al_oracle_selection.run(args)
    elif args.task == 'al_oracle_prediction':
        al_oracle_prediction.run(args)
    elif args.task == 'toy_compute_metrics':
        compute_metrics.run(args)
    else:
        # Extend your experiments here.
        raise NotImplementedError













