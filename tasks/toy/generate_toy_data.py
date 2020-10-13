import sys
import os.path as osp
root_path = osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__))))
sys.path.append(root_path)
import numpy as np
import random

from global_settings.constants import TOY_DATA_PATH
from utils.logging import makedirs
from utils.toy_utils import  compute_cov_relu_kernel


def mkdata_relu(N_train, N_pool, N_test, dim, seed=0, iseed=1, obs_var=1e-2):
    N_data, N_dim = N_train + N_test + N_pool, dim
    np.random.seed(seed)
    random.seed(seed)

    X = np.random.randn(N_data, N_dim)

    Kff = compute_cov_relu_kernel(X)
    L = np.linalg.cholesky(Kff)
    f_X = L @ np.random.normal(0, 1., size=Kff.shape[0])

    noise = np.random.normal(0, obs_var ** 0.5, size=f_X.shape)
    Y = f_X + noise
    indices = list(range(N_train + N_test + N_pool))

    np.random.seed(iseed)
    random.seed(iseed)
    random.shuffle(indices)

    train_X = X[indices[:N_train]]
    train_Y = Y[indices[:N_train]]

    test_X = X[indices[N_train + N_pool:]]
    test_Y = Y[indices[N_train + N_pool:]]

    pool_X = X[indices[N_train:N_train + N_pool]]
    pool_Y = Y[indices[N_train:N_train + N_pool]]

    return train_X, train_Y, pool_X, pool_Y, test_X, test_Y, X, f_X, Kff, None, None, indices


if __name__ == '__main__':
    obs_var = 1e-2
    for dim in [1, 3, 5, 7, 9]:
        N = 5 * dim
        N_pool = 500
        N_test = 200
        for iseed in range(1, 51):
            train_X, train_Y, pool_X, pool_Y, test_X, test_Y, all_X, all_Y, KXX, nkn, sess, indices = mkdata_relu(
                N, N_pool, N_test, dim, 121, iseed, obs_var=obs_var)

            pool_indices = indices[N:N + N_pool]
            test_indices = indices[N + N_pool:]
            indices = indices[:N]

            # pt = list(range(100))
            pt = pool_indices + test_indices
            mu = KXX[pt, :][:, indices] @ np.linalg.inv(KXX[indices, :][:, indices] + obs_var * np.eye(len(indices))) @ \
                 all_Y[indices]
            Kff = KXX[pt, :][:, pt] - KXX[pt, :][:, indices] @ np.linalg.inv(
                KXX[indices, :][:, indices] + obs_var * np.eye(len(indices))) @ KXX[pt, :][:, indices].T

            makedirs(osp.join(root_path, TOY_DATA_PATH, 'reluLarge', 'dim%d_seed%d.npz' % (dim, iseed)))
            np.savez(osp.join(root_path, TOY_DATA_PATH, 'reluLarge', 'dim%d_seed%d.npz' % (dim, iseed)), **{
                'train_x': train_X,
                'train_y': train_Y,
                'test_x': test_X,
                'test_y': test_Y,
                'pool_x': pool_X,
                'pool_y': pool_Y,
                'all_X': all_X,
                'all_f': all_Y,
                'prior_KXX': KXX,
                'obs_var': obs_var,
                'indices': indices,
                'splits_tr_p_te': [N, N_pool, N_test],
                'posterior_mu': mu,
                'posterior_Kff': Kff
            })


