import numpy as np


_VAR_BS, _VAR_WS = [2.5, 2.5], [1.414, 1.737]
def compute_cov_relu_kernel(x, var_bs=_VAR_BS, var_ws=_VAR_WS):
    # x: n * d
    # var = 1 for all layers
    K = var_bs[0] + var_ws[0] * (x @ x.T) / x.shape[1]
    n_hidden_layers = len(var_bs) - 1
    for l in range(n_hidden_layers):
        K = compute_gaussian_relu_expectation(K, var_bs[l + 1], var_ws[l + 1])
    return K


def compute_gaussian_relu_expectation(cov, var_b, var_w):
    # cov N-by-N
    diag_cov = np.diag(cov)
    diag_cov = np.reshape(diag_cov, [-1, 1])  # 1-by-N
    det = np.sqrt(-cov ** 2 + diag_cov @ diag_cov.T) + 1e-10  # N-by-N
    next_cov = 2 * det + cov * np.pi + 2 * cov * np.arctan(cov / det)  # N-by-N
    next_cov = next_cov / (4 * np.pi)
    return next_cov * var_w + var_b