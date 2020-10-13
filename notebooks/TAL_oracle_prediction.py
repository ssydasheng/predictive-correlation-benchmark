
import sys
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import json
sns.set(style="whitegrid")

from global_settings.constants import RESULT_AL_PATH

datasets = ['boston', 'concrete', 'energy', 'wine', 'yacht', 'kin8nm', 'naval', 'power_plant']
methods = ['gp', 'bnn', 'nng', 'hmc', 'gaussian', 'dropout', 'ensemble']
names = ['(SV)GP-RBF', 'BBB', 'NNG', 'HMC', 'FBNN', 'Dropout', 'Ensemble']


def plot_curve(board, x, y, y_bot, y_up, color1, color2, label, linewidth=3, markersize=6):
    if label is not None:
        board.plot(x, y, color=color1, marker=color2[0], linestyle=color2[1], label=label, linewidth=linewidth, markersize=markersize)
    else:
        board.plot(x, y, color=color1, marker=color2[0], linestyle=color2[1], linewidth=linewidth, markersize=markersize)
    board.fill_between(x, y_bot, y_up, color=color1, alpha=0.11)

def get_y_bot_and_up(y, var):
    assert len(y) == len(var)
    indices = list(range(len(y)))
    y_bot = [y[_] - var[_] for _ in indices]
    y_up = [y[_] + var[_] for _ in indices]
    return y_bot, y_up

def pot_subplot_rmse(a, dd, n, label=None, color1=None, color2=None):
    # dd = vis_data[d][c][m]
    rmse, rmse_var, lld, lld_var = dd['rmse_mean'], dd['rmse_var'], dd['lld_mean'], dd['lld_var']
    r_bot, r_up = get_y_bot_and_up(rmse, [_**0.5/n**0.5 for _ in rmse_var])
    l_bot, l_up = get_y_bot_and_up(lld, [_**0.5/n**0.5 for _ in  lld_var])
    x = list(range(len(r_bot)))
    plot_curve(a, x, rmse, r_bot, r_up, color1, color2, label=label)
    
def pot_subplot_lld(a, dd, n, label=None, color1=None, color2=None):
    # dd = vis_data[d][c][m]
    rmse, rmse_var, lld, lld_var = dd['rmse_mean'], dd['rmse_var'], dd['lld_mean'], dd['lld_var']
    r_bot, r_up = get_y_bot_and_up(rmse, [_**0.5/n**0.5 for _ in rmse_var])
    l_bot, l_up = get_y_bot_and_up(lld, [_**0.5/n**0.5 for _ in lld_var])
    x = list(range(len(r_bot)))
    plot_curve(a, x, lld, l_bot, l_up, color1, color2, label=label)

def plot_comparisons_small(methods, method_names, datasets, n_runs=10, s=(-6, -5)):
    def init_plotting():
        #plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["figure.figsize"] = [18.5, 14]
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 2.2 * plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 2.2 * plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 2.2 * plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = 1.2 * plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = 1.1 * plt.rcParams['font.size']
    init_plotting()
    colors = ['g', 'b', 'k', 'r', 'm', 'indigo', 'darkorange'][::-1]
    colors = ['#e874b9', '#b15928', '#6a3d9a', '#33a02c', '#737373', '#e31a1c','#1f78b4']
    
    nrs, ncs = 2, 5
    fig, axs = plt.subplots(nrs, ncs, sharex=True)
    
    # fig.suptitle('Test RMSE and LLD', fontsize=20, y=1.05)
    print(method_names)
    for data_idx in range(len(datasets)):
        r, c = data_idx // ncs, data_idx % ncs
        for method_idx in range(len(method_names)):
            name = method_names[method_idx]
            result_dir = osp.join(RESULT_AL_PATH, '%s_%s_%s.json' % (datasets[data_idx], 'oracle', methods[method_idx]))
            data = json.load(open(result_dir))
            item = 'rmse'
            if data_idx == 0 and item == 'rmse':
                label = name
                print(name)
            else:
                label = None
            a = axs[1, c]
            a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if c == 0:
                a.set_ylabel('$\leftarrow$ RMSE ')
            a.set_xlim([0, 10])
            pot_subplot_rmse(a, 
                             data, 
                             n_runs,
                             label=label, 
                             color1=colors[method_idx], 
                             color2='o-')
    
    for data_idx in range(len(datasets)):
        r, c = data_idx // ncs + nrs//2, data_idx % ncs
        for method_idx in range(len(method_names)):
            name = method_names[method_idx]
            dataset = datasets[data_idx]
            result_dir = osp.join(RESULT_AL_PATH, '%s_%s_%s.json' % (datasets[data_idx], 'oracle', methods[method_idx]))
            data = json.load(open(result_dir))
            item = 'lld'
            if data_idx == 0 and item == 'rmse':
                label = name
            else:
                label = None
            a = axs[0, c]
            a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            if c == 0:
                a.set_ylabel('LLD $\\rightarrow$')
            a.set_title(dataset)
            a.set_xlim([0, 10])
            a.set_xticks([0,2,4,6,8,10])
            pot_subplot_lld(a, 
                            data, 
                            n_runs,
                            label=label, 
                            color1=colors[method_idx], 
                            color2='o-')
    
    lgd = fig.legend(loc='upper center', bbox_to_anchor=(0.525, 1.13),
               fancybox=False, ncol=len(methods)//2+1)
    plt.tight_layout(pad=0.4, w_pad=0.25, h_pad=0.56)
    for line in lgd.get_lines():
        line.set_linewidth(7.0)
    plt.savefig('notebooks/figures/TAL_oracle_small.pdf', bbox_extra_artists=[lgd], bbox_inches="tight")


def plot_comparisons_large(methods, method_names, datasets, n_runs=10, s=(-6, -5)):
    def init_plotting():
        # plt.rcParams['font.family'] = 'Times New Roman'
        plt.rcParams["figure.figsize"] = [16, 9]
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['font.size'] = 16
        plt.rcParams['axes.labelsize'] = 2.2 * plt.rcParams['font.size']
        plt.rcParams['axes.titlesize'] = 2.2 * plt.rcParams['font.size']
        plt.rcParams['legend.fontsize'] = 2.2 * plt.rcParams['font.size']
        plt.rcParams['xtick.labelsize'] = 1.2 * plt.rcParams['font.size']
        plt.rcParams['ytick.labelsize'] = 1.1 * plt.rcParams['font.size']
        plt.rcParams.update({'figure.autolayout': True})

    init_plotting()
    colors = ['g', 'b', 'k', 'r', 'm', 'indigo', 'darkorange'][::-1]
    colors = ['#e874b9', '#b15928', '#6a3d9a', '#33a02c', '#737373', '#e31a1c', '#1f78b4']

    nrs, ncs = 1, 4
    fig, axs = plt.subplots(nrs, ncs)

    for data_idx in range(len(datasets)):
        r, c = 0, data_idx % ncs
        for method_idx in range(len(method_names)):
            name = method_names[method_idx]
            dataset = datasets[data_idx]
            result_dir = osp.join(RESULT_AL_PATH, '%s_%s_%s.json' % (datasets[data_idx], 'oracle', methods[method_idx]))
            data = json.load(open(result_dir))
            item = 'rmse'
            if data_idx == 0 and item == 'rmse':
                label = name
            else:
                label = None
            a = axs[c]
            a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            dstr = 'Kin8nm'
            if dataset == 'power_plant':
                dstr = 'Power_plant'
            a.set_title('%s\n(RMSE$\downarrow$)' % dstr)
            a.set_xlim([0, 10])
            a.set_xticks([0, 2, 4, 6, 8, 10])
            a.set_xlabel('Acquired\ndata(%)')
            pot_subplot_rmse(a,
                             data,
                             n_runs,
                             label=label,
                             color1=colors[method_idx],
                             color2='o-')

    for data_idx in range(len(datasets)):
        r, c = 0, data_idx % ncs + 2
        for method_idx in range(len(method_names)):
            name = method_names[method_idx]
            dataset = datasets[data_idx]
            result_dir = osp.join(RESULT_AL_PATH, '%s_%s_%s.json' % (datasets[data_idx], 'oracle', methods[method_idx]))
            data = json.load(open(result_dir))
            item = 'lld'
            if data_idx == 0 and item == 'rmse':
                label = name
            else:
                label = None
            a = axs[c]
            a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            dstr = 'Kin8nm'
            if dataset == 'power_plant':
                dstr = 'Power_plant'
            a.set_title('%s\n(LLD$\\uparrow$)' % dstr)
            a.set_xlim([0, 10])
            a.set_xlabel('Acquired\ndata(%)')
            a.set_xticks([0, 2, 4, 6, 8, 10])
            pot_subplot_lld(a,
                            data,
                            n_runs,
                            label=label,
                            color1=colors[method_idx],
                            color2='o-')

    plt.tight_layout(pad=0.4, w_pad=0.25, h_pad=0.56)
    plt.savefig('./mean_marginal_large_rmse_lld_ndrp.pdf', bbox_inches="tight")

if __name__ == '__main__':
    plot_comparisons_small(methods[::-1], names[::-1], datasets[:5], 10, s=(-6, -5))
    plot_comparisons_large(methods[::-1], names[::-1], datasets[5:7], 10, s=(-6, -5))
