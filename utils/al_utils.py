import numpy as np
import random
from data.data import (uci_woval, standardize)
from easydict import EasyDict as edict
from scipy import linalg as scipy_linalg
from sklearn.model_selection import train_test_split


def HParams(**kwargs):
    ed = edict()
    for name, value in kwargs.items():
        ed[name] = value
    return ed


def compute_corr_or_covar(corr_op, model, sess, data, n_particles=5000):
    pool_x, test_x = data.pool_x, data.test_x

    N_pool, N_test = pool_x.shape[0], test_x.shape[0]

    x = np.concatenate([pool_x, test_x], 0)
    # with tf.device('/cpu:0'):
    print("N_particles: %d" % n_particles)
    feed_dict = {model.x_pred: x, model.x: data.train_x, model.y: data.train_y, model.n_particles: n_particles}
    corr = sess.run([corr_op], feed_dict=feed_dict)
    return corr, N_pool, N_test


def get_argmax_corr_square(corr, n_pool, n_test, topk=1):
    assert n_pool + n_test == corr.shape[0]
    sub_mat = corr[:n_pool, n_pool:]**2. #TODO
    sub_mat = -np.sum(np.log(1-sub_mat), axis=1)
    indices = np.argsort(-np.squeeze(sub_mat))
    return indices.tolist()[:topk]
    # return np.argmax(sub_mat)


def get_argmax_var(cov, n_pool, n_test, topk=1):
    assert n_pool + n_test == cov.shape[0]
    sub_vec = np.diag(cov)[:n_pool]
    indices = np.argsort(-np.squeeze(sub_vec))
    return indices.tolist()[:topk]


def get_batch_mode_MMIG_idx(cov, selected_indices, pool_indices, test_indices):
    info_gain = []
    cov_tt = cov[test_indices][..., test_indices]
    for idx, p in enumerate(pool_indices):
        cov_qq = cov[selected_indices+[p]][:, selected_indices+[p]]
        cov_qq_inv = np.linalg.inv(cov_qq)
        cov_qt = cov[selected_indices+[p]][..., test_indices]
        ifgs = np.sum(np.multiply(cov_qt.T, (cov_qq_inv @ cov_qt).T), 1)  # t/t
        ifgs = np.squeeze(ifgs) / np.squeeze(np.diag(cov_tt))
        ifgs = -np.sum(np.log(1-ifgs))
        info_gain.append(ifgs)
    return np.argmax(info_gain)


def get_batch_mode_MMIG_idx_v2(cov, selected_indices, pool_indices, test_indices):
    info_gain = []
    for idx, p in enumerate(pool_indices):
        print('%d/%d.' % (idx, len(pool_indices)))
        cov_tt = cov[test_indices][..., test_indices]
        cov_qq = cov[selected_indices + [p]][..., selected_indices + [p]]
        cov_qq_inv = np.linalg.inv(cov_qq)
        cov_qt = cov[selected_indices + [p]][..., test_indices]
        ifgs = cov_qt.T @ cov_qq_inv @ cov_qt  # t/t
        ifgs = np.squeeze(np.diag(ifgs)) / np.squeeze(np.diag(cov_tt))
        ifgs = -np.sum(np.log(1 - ifgs))
        info_gain.append(ifgs)
    return np.argmax(info_gain)


def get_batch_mode_TIG_idx_Fast(cov, selected_indices, pool_indices):
    pass


def get_batch_mode_MMIG_idx_Fast(cov, selected_indices, pool_indices, test_indices):
    eps = 1e-6
    if len(selected_indices) == 0:  # dummy node
        cov_st = np.reshape(np.array([0.] * len(test_indices)), [1, len(test_indices)])
        cov_sq = np.reshape(np.array([0.] * len(pool_indices)), [1, len(pool_indices)])
    else:
        cov_st = cov[selected_indices][:, test_indices]
        cov_sq = cov[selected_indices][:, pool_indices]

    cov_tt = cov[test_indices][:, test_indices]
    cov_qt = cov[pool_indices][:, test_indices]

    if len(selected_indices) == 0:
        L = np.array([[1.0]])
    else:
        L = np.linalg.cholesky(cov[selected_indices][:, selected_indices]) # [n_selected, n_selected]

    Linv_Kst = scipy_linalg.solve_triangular(L, cov_st, lower=True) # [n_selected, n_test]
    Linv_Kst_2norm = np.linalg.norm(Linv_Kst, axis=0)**2. # [n_test]

    Linv_Kps = scipy_linalg.solve_triangular(L, cov_sq, lower=True) #[n_selected, n_pool]
    Linv_Kps_2norm = np.linalg.norm(Linv_Kps, axis=0)**2. #[n_pool]
    E = np.sqrt(np.diagonal(cov)[pool_indices] - Linv_Kps_2norm) # [n_pool]
    assert np.all(E > 0), 'E = {}'.format(E)

    F = -scipy_linalg.solve_triangular(L.T, Linv_Kps, lower=False) / np.expand_dims(E, 0) # [n_selected, n_pool]
    V = F.T @ cov_st + cov_qt / np.expand_dims(E, 1) # [n_pool, n_test]

    corr_square = (V**2. + Linv_Kst_2norm) / np.diagonal(cov_tt) # [n_pool, n_test]
    # return (corr_square[0])
    # assert np.all(corr_square < 1. + eps), 'corr_square = {}'.format(corr_square)
    info_gain = - np.sum(np.log(1. + eps-corr_square), 1) # [n_pool]
    return np.argmax(info_gain)
    #
    # for idx, p in enumerate(pool_indices):
    #     print('%d/%d.' % (idx, len(pool_indices)))
    #
    #     cov_ps = column_selected[p] #[n_s]
    #     pp = cov[p][p]
    #
    #     # the cholesky of cov_{s,p} is Lq = [[L, 0], [d^t, e]]. d=L^{-1}K_{sq}, e= K_{qq} - d^t d.
    #     d = scipy_linalg.solve_triangular(L, cov_ps)
    #     e = cov[p][p] - Linv_Kps[p]
    #     assert e > 0
    #
    #     # K_{sq, t}^{t} K_{qq}^{-1} K_{sq, t} = K_{sq, t}^t (Lq Lq^{t})^{-1} K_{sq, t} = U^{t}U; U=Lq^{-1}K_{sq, t}
    #     # Lq^{-1} = [[L, 0], [d^t, e]]^{-1} = [[L^{-1}, 0], [f^t, 1/e]]; f= - (L^t)^{-1}d / e
    #     # ...=Lq^{-1}K_{sq, t}=[L^{-1}K_{st}, f^t K_{st} + K_{qt}/e]. Let V=f^t K_{st} + K_{qt}/e
    #     corr = (L^{-1}K_{st})^2 + V^2
    #
    #     cov_qq = cov[selected_indices+[p]][..., selected_indices+[p]]
    #     cov_qq_inv = np.linalg.inv(cov_qq)
    #     cov_qt = cov[selected_indices+[p]][..., test_indices]
    #     ifgs = np.sum(np.multiply(cov_qt, (cov_qq_inv @ cov_qt)), 0) # t/t
    #     ifgs = np.squeeze(ifgs) / np.squeeze(np.diag(cov_tt))
    #     ifgs = -np.sum(np.log(1-ifgs))
    #     info_gain.append(ifgs)
    # return np.argmax(info_gain)


# ========================= for active learning data processing ==============================
def fetch_active_learning_dataset(args, seed):
    split_seed = seed
    seed = 8888
    dataset = uci_woval(args.dataset, seed=seed, standardization=False, test_size=args.test_ratio)
    train_x_all, test_x, train_y_all, test_y = dataset.x_train, dataset.x_test, dataset.y_train, dataset.y_test

    train_x, valid_x, train_y, valid_y = train_test_split(train_x_all, train_y_all, test_size=0.2, random_state=seed)
    ind = np.arange(train_x.shape[0] + test_x.shape[0])
    np.random.seed(split_seed)
    np.random.shuffle(ind)
    n = train_x.shape[0]
    X_all, Y_all = np.concatenate([train_x, test_x], 0), np.concatenate([train_y, test_y])
    train_x, train_y = X_all[ind[:n]], Y_all[ind[:n]]
    test_x, test_y = X_all[ind[n:]], Y_all[ind[n:]]
    train_x_all = np.concatenate([train_x, valid_x], 0)
    train_y_all = np.concatenate([train_y, valid_y])

    train_x_all, test_x, _, _ = standardize(train_x_all, test_x)

    n_train = int((train_x_all.shape[0] + test_x.shape[0]) * args.train_ratio)
    train_x = train_x_all[:n_train]
    pool_x = train_x_all[n_train:]

    train_y = train_y_all[:n_train]
    pool_y = train_y_all[n_train:]
    return HParams(train_x=train_x, train_y=train_y,
                   pool_x=pool_x, pool_y=pool_y,
                   test_x=test_x, test_y=test_y)


def normalize_active_learning_data(data, has_pool=True):
    if has_pool:
        train_y, test_y, pool_y = data.train_y, data.test_y, data.pool_y
        new_train_y, new_test_y, new_pool_y, y_mean, y_std = standardize(train_y, test_y, pool_y)
        return HParams(train_x=data.train_x, train_y=new_train_y,
                       pool_x=data.pool_x, pool_y=new_pool_y,
                       test_x=data.test_x, test_y=new_test_y,
                       mean_y_train=y_mean, std_y_train=y_std)
    else:
        train_y, test_y = data.train_y, data.test_y
        new_train_y, new_test_y, y_mean, y_std = standardize(train_y, test_y)
        return HParams(train_x=data.train_x, train_y=new_train_y,
                       test_x=data.test_x, test_y=new_test_y,
                       mean_y_train=y_mean, std_y_train=y_std)


def update_dataset(original_dataset, data_idxs):
    dataset = original_dataset
    train_x, test_x, train_y, test_y = dataset.train_x, dataset.test_x, dataset.train_y, dataset.test_y
    pool_x, pool_y = dataset.pool_x, dataset.pool_y

    new_data_x = pool_x[data_idxs]
    new_data_y = pool_y[data_idxs]

    pool_indices = list(set(range(pool_x.shape[0])) - set(data_idxs))

    train_x = np.concatenate([train_x, new_data_x], 0)
    train_y = np.concatenate([train_y, new_data_y], 0)

    pool_x = pool_x[pool_indices]  # np.concatenate([pool_x[:data_idx], pool_x[data_idx+1:]], 0)
    pool_y = pool_y[pool_indices]  # np.concatenate([pool_y[:data_idx], pool_y[data_idx+1:]], 0)

    return HParams(train_x=train_x, train_y=train_y,
                   pool_x=pool_x, pool_y=pool_y,
                   test_x=test_x, test_y=test_y)


def get_selected_data_idxs(train_x, test_x, pool_x, dataset, args, covar_op, corr_op, model, sess, n_particles=1000):
    num_points_to_collect = max(int((train_x.shape[0] + test_x.shape[0] + pool_x.shape[0]) * args.active_ratio), 1)

    if args.criteria == 'tig':
        print("Using: total_info_gain")
        corr_or_covar, N_pool, N_test = compute_corr_or_covar(covar_op, model, sess, dataset, n_particles=n_particles)
        data_idxs = get_argmax_var(corr_or_covar[0], N_pool, N_test, num_points_to_collect)
    elif args.criteria == 'mig':
        print("Using [MIG]: mean_marginal_info_gain")
        corr_or_covar, N_pool, N_test = compute_corr_or_covar(corr_op, model, sess, dataset, n_particles=n_particles)
        data_idxs = get_argmax_corr_square(corr_or_covar[0], N_pool, N_test, num_points_to_collect)
    elif args.criteria == 'random':
        print("Using: random")
        indices = list(range(pool_x.shape[0]))
        random.shuffle(indices)
        data_idxs = indices[:num_points_to_collect]
    elif args.criteria == 'batchMIG':
        print("[Greedy-BatchMMIG] Using: mean_marginal_info_gain")
        corr_or_covar, N_pool, N_test = compute_corr_or_covar(covar_op, model, sess, dataset, n_particles=n_particles)
        data_idxs = []
        pool_indices = list(range(pool_x.shape[0]))
        test_indices = list(range(pool_x.shape[0], pool_x.shape[0] + test_x.shape[0]))
        for idx in range(num_points_to_collect):
            b_idx = get_batch_mode_MMIG_idx_Fast(corr_or_covar[0], data_idxs, pool_indices, test_indices)
            data_idxs.append(pool_indices[b_idx])
            pool_indices = pool_indices[:b_idx] + pool_indices[b_idx + 1:]
    else:
        raise NotImplementedError('%s is not implemented.' % args.criteria)
    return data_idxs



