from easydict import EasyDict as edict

GLOBAL_BLOB = edict(
    supported_methods=['mfbnn',  # FBNN
                       'fbnn',  # FBNN
                       'bbb',  # BBB
                       'gp',  # GP
                       'dropout',  # Dropout
                       'ensemble',  # Ensemble
                       'homo_ensemble',  # Ensemble
                       'svgp',  # SVGP
                       'nng',  # NNG
                       'hmc',  # HMC
                       'emhmc'],  # EM-HMC

)


TOY_DATA_PATH = 'results/toy/data'
# RESULT_TOY_METRIC_PATH = 'results/toy/metric'
RESULT_TOY_METRIC_PATH = '/scratch/hdd001/home/ssy/ScalableFBNN/Oct11/metrics'
LOG_TOY_METRIC_PATH = 'logs/toy/metric'

RESULT_REG_PATH = 'results/regression'
LOG_REG_PATH = 'logs/regression'

ORACLE_CKPT_DIR = 'results/AL/oracle_ckpt'

RESULT_AL_PATH = 'results/AL/'
RESULT_AL_SELECTION_PATH = 'results/AL/selection'
LOG_AL_SELECTION_PATH = 'logs/AL/selection'
LOG_AL_PREDICTION_PATH = 'logs/AL/prediction'
LOG_AL_PRETRAIN_ORACLE_PATH = 'logs/AL/pretrain'