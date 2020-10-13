import os
import sys
root_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(root_path)
import numpy as np
import random
from sklearn.model_selection import train_test_split

from .hparams import HParams
from .register import register


data_path = os.path.join(root_path, 'data')
DATASETS = dict(
    boston='housing.data',
    concrete='concrete.data',
    energy='energy.data',
    kin8nm='kin8nm.data',
    naval='naval.data',
    power_plant='power_plant.data',
    wine='wine.data',
    yacht='yacht_hydrodynamics.data',
)


@register('uci_woval')
def uci_woval(dataset_name, seed=1, standardization=True, test_size=0.2, train_size=None):
    seed=8888
    data = np.loadtxt(os.path.join(data_path, 'uci', DATASETS[dataset_name]))
    x, y = data[:, :-1], data[:, -1]

    if test_size > 0.:
        x_t, x_v, y_t, y_v = train_test_split(x, y, test_size=test_size, random_state=seed)
    else:
        x_t, x_v, y_t, y_v = x, None, y, None

    if train_size is not None:
        end = int(x.shape[0] * train_size)
        assert end <= x_t.shape[0]
        x_t, y_t = x_t[:end], y_t[:end]

    if standardization:
        x_t, x_v, x_mean, x_std = standardize(x_t, x_v)
        y_t, y_v, y_mean, train_std = standardize(y_t, y_v)
        hparams = HParams(
            x_train=x_t,
            x_test=x_v,
            y_train=y_t,
            y_test=y_v,
            std_y_train=train_std,
            std_x_train=x_std,
            mean_x_train=x_mean,
            mean_y_train=y_mean
        )
    else:
        train_std = -1
        hparams = HParams(
            x_train=x_t,
            x_test=x_v,
            y_train=y_t,
            y_test=y_v,
            std_y_train=train_std
        )

    return hparams


def standardize(data_train, *args):
    """
    Standardize a dataset to have zero mean and unit standard deviation.
    :param data_train: 2-D Numpy array. Training data.
    :param data_test: 2-D Numpy array. Test data.
    :return: (train_set, test_set, mean, std), The standardized dataset and
        their mean and standard deviation before processing.
    """
    std = np.std(data_train, 0, keepdims=True)
    std[std == 0] = 1
    mean = np.mean(data_train, 0, keepdims=True)
    data_train_standardized = (data_train - mean) / std
    output = [data_train_standardized]
    for d in args:
        dd = (d - mean) / std if d is not None else d
        output.append(dd)
    output.append(mean)
    output.append(std)
    return output

