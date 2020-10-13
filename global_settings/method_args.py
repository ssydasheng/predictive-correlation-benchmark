def _bbb_args(main_parser):
    parser = main_parser.add_argument_group('methods')
    parser.add_argument('-nh', '--n_hidden', type=int, default=1,
                        help='How many layers in the network. (default: 1)')
    parser.add_argument('-nu', '--n_units', type=int, default=400,
                        help='How many neurons in each hidden layer. (default: 400)')
    parser.add_argument('-mc', '--marginal_coeff', type=float, default=1.,
                        help='The magrinal coefficient')
    parser.add_argument('--train_samples', type=int, default=10,
                        help='How many samples used for training.')
    parser.add_argument('-kae', '--kl_ann_epochs', type=int, default=1)
    parser.add_argument('-act', '--act', type=str, default='relu')
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_cov_samples', type=int, default=1000)

    parser.add_argument('--opt_eta', action='store_true')
    parser.add_argument('--init_eta', type=float, default=1.)

def _dropout_args(main_parser):
    parser = main_parser.add_argument_group('methods')
    parser.add_argument('-nh', '--n_hidden', type=int, default=1,
                        help='How many layers in the network. (default: 1)')
    parser.add_argument('-nu', '--n_units', type=int, default=400,
                        help='How many neurons in each hidden layer. (default: 400)')
    parser.add_argument('-act', '--act', type=str, default='relu')
    parser.add_argument('--dropout_tau', type=float, default=0.025)
    parser.add_argument('--dropout_rate', type=float, default=0.005)
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_cov_samples', type=int, default=1000)
    parser.add_argument('--train_samples', type=int, default=10,
                        help='How many samples used for training.')


def _emhmc_args(main_parser):
    parser = main_parser.add_argument_group('methods')
    parser.add_argument('-nh', '--n_hidden', type=int, default=1,
                        help='How many layers in the network. (default: 1)')
    parser.add_argument('-nu', '--n_units', type=int, default=400,
                        help='How many neurons in each hidden layer. (default: 400)')
    parser.add_argument('-act', '--act', type=str, default='relu')
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_cov_samples', type=int, default=1000)
    parser.add_argument('--train_samples', type=int, default=10,
                        help='How many samples used for training.')

    parser.add_argument('--hmc_burnin', type=int, default=10000)
    parser.add_argument('--hmc_samples', type=int, default=100, help="The number of samples per chain")
    parser.add_argument('--hmc_num_chains', type=int, default=10)
    parser.add_argument('--hmc_leap_frog_steps', type=int, default=6)
    parser.add_argument('--hmc_step_size', type=float, default=0.01)
    parser.add_argument('--num_steps_between_samples', type=int, default=100)
    parser.add_argument('--hmc_n_steps_per_em', type=int, default=10)
    parser.add_argument('--hmc_n_steps_adapt_step_size', type=int, default=10)

    parser.add_argument('--opt_eta', action='store_true')
    parser.add_argument('--init_eta', type=float, default=1.)

def _ensemble_args(main_parser):
    parser = main_parser.add_argument_group('methods')
    parser.add_argument('-nh', '--n_hidden', type=int, default=1,
                        help='How many layers in the network. (default: 1)')
    parser.add_argument('-nu', '--n_units', type=int, default=400,
                        help='How many neurons in each hidden layer. (default: 400)')
    parser.add_argument('-act', '--act', type=str, default='relu')
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_cov_samples', type=int, default=1000)
    parser.add_argument('--train_samples', type=int, default=10,
                        help='How many samples used for training.')
    parser.add_argument('-n_networks', '--num_networks', type=int, default=5)


def _fbnn_args(main_parser):
    parser = main_parser.add_argument_group('methods')
    parser.add_argument('-nh', '--n_hidden', type=int, default=1,
                        help='How many layers in the network. (default: 1)')
    parser.add_argument('-nu', '--n_units', type=int, default=400,
                        help='How many neurons in each hidden layer. (default: 400)')
    parser.add_argument('-act', '--act', type=str, default='relu')
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_cov_samples', type=int, default=1000)
    parser.add_argument('--train_samples', type=int, default=10,
                        help='How many samples used for training.')
    parser.add_argument('--gpnet', type=str, default='rf')
    parser.add_argument('-nr', '--n_rand', type=int, default=100)
    parser.add_argument('-kern', '--kernel', type=str, default='rbf')
    parser.add_argument('--n_eigen_threshold', type=float, default=0.99)
    parser.add_argument('-ard', '--ARD', action='store_true')


def _gp_args(main_parser):
    parser = main_parser.add_argument_group('methods')
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_cov_samples', type=int, default=1000)
    parser.add_argument('--train_samples', type=int, default=10,
                        help='How many samples used for training.')
    parser.add_argument('-kern', '--kernel', type=str, default='rbf')
    parser.add_argument('-ard', '--ARD', action='store_true')
    # parser.add_argument('-nb', '--n_base', type=int, default=1000)

def _hmc_args(main_parser):
    parser = main_parser.add_argument_group('methods')
    parser.add_argument('-nh', '--n_hidden', type=int, default=1,
                        help='How many layers in the network. (default: 1)')
    parser.add_argument('-nu', '--n_units', type=int, default=400,
                        help='How many neurons in each hidden layer. (default: 400)')
    parser.add_argument('-act', '--act', type=str, default='relu')
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_cov_samples', type=int, default=1000)
    parser.add_argument('--train_samples', type=int, default=10,
                        help='How many samples used for training.')

    parser.add_argument('--hmc_burnin', type=int, default=10000)
    parser.add_argument('--hmc_samples', type=int, default=100, help="The number of samples per chain")
    parser.add_argument('--hmc_num_chains', type=int, default=10)
    parser.add_argument('--hmc_leap_frog_steps', type=int, default=6)
    parser.add_argument('--hmc_step_size', type=float, default=0.01)
    parser.add_argument('--num_steps_between_samples', type=int, default=100)
    parser.add_argument('--hmc_n_steps_per_em', type=int, default=10)
    parser.add_argument('--hmc_n_steps_adapt_step_size', type=int, default=10)

    parser.add_argument('--opt_eta', action='store_true')
    parser.add_argument('--init_eta', type=float, default=1.)


def _mfbnn_args(parser):
    return _fbnn_args(parser)


def _nng_args(main_parser):
    parser = main_parser.add_argument_group('methods')
    parser.add_argument('-nh', '--n_hidden', type=int, default=1,
                        help='How many layers in the network. (default: 1)')
    parser.add_argument('-nu', '--n_units', type=int, default=400,
                        help='How many neurons in each hidden layer. (default: 400)')
    parser.add_argument('-mc', '--marginal_coeff', type=float, default=1.,
                        help='The magrinal coefficient')
    parser.add_argument('--train_samples', type=int, default=10,
                        help='How many samples used for training.')
    parser.add_argument('-kae', '--kl_ann_epochs', type=int, default=1)
    parser.add_argument('-act', '--act', type=str, default='relu')
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_cov_samples', type=int, default=1000)

    parser.add_argument('--opt_eta', action='store_true')
    parser.add_argument('--init_eta', type=float, default=1.)

def _svgp_args(main_parser):
    parser = main_parser.add_argument_group('methods')
    parser.add_argument('--test_samples', type=int, default=100)
    parser.add_argument('--eval_cov_samples', type=int, default=1000)
    parser.add_argument('--train_samples', type=int, default=10,
                        help='How many samples used for training.')
    parser.add_argument('-kern', '--kernel', type=str, default='rbf')


def load_args(parser, argv):
    method_idx = argv.index('--method')
    method = argv[method_idx + 1]
    binding = {
        'bbb': _bbb_args,
        'dropout': _dropout_args,
        'emhmc': _emhmc_args,
        'ensemble': _ensemble_args,
        'homo_ensemble': _ensemble_args,
        'fbnn': _fbnn_args,
        'gp': _gp_args,
        'hmc': _hmc_args,
        'mfbnn': _mfbnn_args,
        'nng': _nng_args,
        'svgp': _svgp_args
    }

    binding[method](parser)