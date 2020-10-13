import sys
import os
import os.path as osp
sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import FormatStrFormatter
import json

from global_settings.constants import RESULT_AL_PATH

sns.set(style="whitegrid")


def init_plotting():
    plt.rcParams["figure.figsize"] = [16, 10]
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['font.size'] = 16
    plt.rcParams['axes.labelsize'] = 1.7 * plt.rcParams['font.size']
    plt.rcParams['axes.titlesize'] = 1.7 * plt.rcParams['font.size']
    plt.rcParams['legend.fontsize'] = 1.8 * plt.rcParams['font.size']
    plt.rcParams['xtick.labelsize'] = 1.2 * plt.rcParams['font.size']
    plt.rcParams['ytick.labelsize'] = 1.2 * plt.rcParams['font.size']


init_plotting()

START = 0
def plot_curve(board, x, y, y_bot, y_up, color1, color2, label, linewidth=2, markersize=6):
    if label is not None:
        board.plot(x[START:], y[START:], color=color1, marker=color2[0], linestyle=color2[1], label=label, linewidth=linewidth,
                   markersize=markersize)
    else:
        board.plot(x[START:], y[START:], color=color1, marker=color2[0], linestyle=color2[1], linewidth=linewidth,
                   markersize=markersize)
    board.fill_between(x[START:], y_bot[START:], y_up[START:], color=color1, alpha=0.2)

def get_y_bot_and_up(y, var):
    assert len(y) == len(var)
    indices = list(range(len(y)))
    y_bot = [y[_] - var[_] for _ in indices]
    y_up = [y[_] + var[_] for _ in indices]
    return y_bot, y_up

def pot_subplot_rmse(a, dd, n=10, label=None, color1=None, color2=None):
    # dd = vis_data[d][c][m]
    rmse, rmse_var, lld, lld_var = dd['rmse_mean'], dd['rmse_var'], dd['lld_mean'], dd['lld_var']
    r_bot, r_up = get_y_bot_and_up(rmse, [_**0.5/n**0.5 for _ in rmse_var])
    l_bot, l_up = get_y_bot_and_up(lld, [_**0.5/n**0.5 for _ in lld_var])
    x = list(range(len(r_bot)))
    plot_curve(a, x, rmse, r_bot, r_up, color1, color2, label=label)

def pot_subplot_lld(a, dd, n=10, label=None, color1=None, color2=None):
    # dd = vis_data[d][c][m]
    rmse, rmse_var, lld, lld_var = dd['rmse_mean'], dd['rmse_var'], dd['lld_mean'], dd['lld_var']
    r_bot, r_up = get_y_bot_and_up(rmse, [_**0.5/n**0.5 for _ in rmse_var])
    l_bot, l_up = get_y_bot_and_up(lld, [_**0.5/n**0.5 for _ in lld_var])
    x = list(range(len(r_bot)))
    plot_curve(a, x, lld, l_bot, l_up, color1, color2, label=label)

def fetch_results(criterion, dataset, cond=lambda x,y: True):
    path = osp.join(RESULT_AL_PATH, '%s_%s_%s.json' % (dataset, 'oracle', 'oracle'))
    res = json.load(open(path))
    return res

def plot(s=(-6, -5)):
    method, method_name = 'oracle', 'oracle'
    datasets = ['boston', 'concrete', 'energy', 'wine', 'yacht', 'kin8nm', 'naval', 'power_plant']
    res = {}
    for d in datasets:
        print(d)
        res[d] = {}
        for c in ['batch', 'mean', 'total', 'random']:
            print(c)
            res[d][c] = fetch_results(c, d)

    plot_rmse_all(res, method, method_name, s)
    plot_lld_all(res, method, method_name)

def plot_rmse_all(src, method, method_name, s=None):
    xlim = [0, 10]
    c1, c2, c3, c4 = '#d7191c', '#2b83ba', '#4dac26', '#ff7f00'
    fig, ax = plt.subplots(2, 4, sharex=True)

    a = ax[0, 0]
    d = 'boston'
    if method == 'gp' and 'rbf' in method_name.lower():
        a.set_ylim(-3.5, -2.5)
    c = 'batch'
    pot_subplot_rmse(a, src[d][c], color1=c1, color2='o-', label='BatchMIG')
    c = 'mean'
    pot_subplot_rmse(a, src[d][c], color1=c2, color2='o-', label='MIG')
    c = 'total'
    pot_subplot_rmse(a, src[d][c], color1=c3, color2='o-', label='TIG')
    c = 'random'
    pot_subplot_rmse(a, src[d][c], color1=c4, color2='o-', label='Random')
    a.set_title(d)
    a.set_ylabel('RMSE')
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    a = ax[0, 1]
    d = 'concrete'
    c = 'batch'
    pot_subplot_rmse(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_rmse(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_rmse(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_rmse(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    a = ax[0, 2]
    d = 'energy'
    c = 'batch'
    pot_subplot_rmse(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_rmse(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_rmse(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_rmse(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    a = ax[0, 3]
    d = 'wine'
    c = 'batch'
    pot_subplot_rmse(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_rmse(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_rmse(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_rmse(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    a = ax[1, 0]
    d = 'yacht'
    c = 'batch'
    pot_subplot_rmse(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_rmse(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_rmse(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_rmse(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a.set_xlim(*xlim)
    a.set_ylabel('RMSE')
    a.set_xlabel('Acquired \n data(%)')
    a.set_xticks([0, 2, 4, 6, 8, 10])
    if method == 'gp' and 'rbf' in method_name.lower():
        a.set_ylim([-4, 0])

    a = ax[1, 1]
    d = 'kin8nm'
    c = 'batch'
    pot_subplot_rmse(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_rmse(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_rmse(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_rmse(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a.set_xlim(*xlim)
    a.set_xlabel('Acquired \n data(%)')
    a.set_xticks([0, 2, 4, 6, 8, 10])

    a = ax[1, 2]
    d = 'naval'
    c = 'batch'
    pot_subplot_rmse(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_rmse(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_rmse(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_rmse(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a.set_xlim(*xlim)
    a.set_xlabel('Acquired \n data(%)')
    a.set_xticks([0, 2, 4, 6, 8, 10])

    a = ax[1, 3]
    d = 'power_plant'
    c = 'batch'
    pot_subplot_rmse(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_rmse(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_rmse(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_rmse(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a.set_xlim(*xlim)
    a.set_xlabel('Acquired \n data(%)')
    a.set_xticks([0, 2, 4, 6, 8, 10])

    lgd = fig.legend(loc='upper center', bbox_to_anchor=(0.52, 1.10),
                     fancybox=False, ncol=4)
    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.5)
    for line in lgd.get_lines():
        line.set_linewidth(7.0)

    # fig.tight_layout()
    plt.savefig('notebooks/figures/%s_rmse.pdf' % method_name,  bbox_extra_artists=[lgd], bbox_inches='tight')


def plot_lld_all(src, method, method_name):
    xlim = [0, 10]
    c1, c2, c3, c4 = '#d7191c', '#2b83ba', '#4dac26', '#ff7f00'
    fig, ax = plt.subplots(2, 4, sharex=True)

    a = ax[0, 0]
    d = 'boston'
    if method == 'gp' and 'rbf' in method_name.lower():
        a.set_ylim(-3.5, -2.5)
    c = 'batch'
    pot_subplot_lld(a, src[d][c], color1=c1, color2='o-', label='BatchMIG')
    c = 'mean'
    pot_subplot_lld(a, src[d][c], color1=c2, color2='o-', label='MIG')
    c = 'total'
    pot_subplot_lld(a, src[d][c], color1=c3, color2='o-', label='TIG')
    c = 'random'
    pot_subplot_lld(a, src[d][c], color1=c4, color2='o-', label='Random')
    a.set_title(d)
    a.set_ylabel('LLD')
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    a = ax[0, 1]
    d = 'concrete'
    c = 'batch'
    pot_subplot_lld(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_lld(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_lld(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_lld(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)

    a = ax[0, 2]
    d = 'energy'
    c = 'batch'
    pot_subplot_lld(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_lld(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_lld(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_lld(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    a = ax[0, 3]
    d = 'wine'
    c = 'batch'
    pot_subplot_lld(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_lld(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_lld(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_lld(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    a = ax[1, 0]
    d = 'yacht'
    c = 'batch'
    pot_subplot_lld(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_lld(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_lld(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_lld(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a.set_xlim(*xlim)
    a.set_ylabel('LLD')
    a.set_xlabel('Acquired \n data(%)')
    a.set_xticks([0, 2, 4, 6, 8, 10])
    if method == 'gp' and 'rbf' in method_name.lower():
        a.set_ylim([-4, 0])

    a = ax[1, 1]
    d = 'kin8nm'
    c = 'batch'
    pot_subplot_lld(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_lld(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_lld(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_lld(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a.set_xlim(*xlim)
    a.set_xlabel('Acquired \n data(%)')
    a.set_xticks([0, 2, 4, 6, 8, 10])

    a = ax[1, 2]
    d = 'naval'
    c = 'batch'
    pot_subplot_lld(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_lld(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_lld(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_lld(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a.set_xlim(*xlim)
    a.set_xlabel('Acquired \n data(%)')
    a.set_xticks([0, 2, 4, 6, 8, 10])

    a = ax[1, 3]
    d = 'power_plant'
    c = 'batch'
    pot_subplot_lld(a, src[d][c], color1=c1, color2='o-')
    c = 'mean'
    pot_subplot_lld(a, src[d][c], color1=c2, color2='o-')
    c = 'total'
    pot_subplot_lld(a, src[d][c], color1=c3, color2='o-')
    c = 'random'
    pot_subplot_lld(a, src[d][c], color1=c4, color2='o-')
    a.set_title(d)
    a.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    a.set_xlim(*xlim)
    a.set_xlabel('Acquired \n data(%)')
    a.set_xticks([0, 2, 4, 6, 8, 10])

    lgd = fig.legend(loc='upper center', bbox_to_anchor=(0.52, 1.10),
                     fancybox=False, ncol=4)
    plt.tight_layout(pad=0.4, w_pad=0.2, h_pad=0.5)
    for line in lgd.get_lines():
        line.set_linewidth(7.0)

    # fig.tight_layout()
    plt.savefig('notebooks/figures/%s_lld.pdf' % method_name,  bbox_extra_artists=[lgd], bbox_inches='tight')


if __name__ == '__main__':
    plot()
