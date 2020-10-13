"""
Each experiment has a respective args function.
"""


def _regression_args(main_parser):
    parser = main_parser.add_argument_group('tasks')
    parser.add_argument('-return_val', '--return_val', action='store_true')
    parser.add_argument('-ckpt_dir', '--ckpt_dir', type=str, default='')


def _transductive_active_learning_args(main_parser):
    parser = main_parser.add_argument_group('tasks')
    parser.add_argument('-oracle_dir', '--oracle_dir', type=str, default='')
    parser.add_argument('-ai', '--active_iterations', type=int, default=10)
    parser.add_argument('--active_ratio', type=float, default=0.01)
    parser.add_argument('-crit', '--criteria', type=str, default='BatchMIG')
    parser.add_argument('--base_method', type=str, default='')


def _toy_args(main_parser): #TODO: methods have different args in toy
    parser = main_parser.add_argument_group('tasks')
    parser.add_argument('--input_dim', type=int)
    parser.add_argument('-crit', '--criteria', type=str, default='BatchMIG')
    parser.add_argument('-ai', '--active_iterations', type=int, default=10)
    parser.add_argument('--active_ratio', type=float, default=0.01)

def load_args(parser, argv):
    method_idx = argv.index('--task')
    method = argv[method_idx + 1]
    binding = {
        'regression': _regression_args,

        'al_selection': _transductive_active_learning_args,
        'al_prediction': _transductive_active_learning_args,
        'al_oracle_pretrain': _transductive_active_learning_args,
        'al_oracle_selection': _transductive_active_learning_args,
        'al_oracle_prediction': _transductive_active_learning_args,

        'toy_compute_metrics': _toy_args,
    }

    binding[method](parser)