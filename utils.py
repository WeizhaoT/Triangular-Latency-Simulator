""" 
utils.py

A tool box for simulator, consisting of various most-used methods
"""
import os
import time
import random
import shutil
import pickle
import platform
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from tqdm import tqdm
from os.path import isdir, exists, join, abspath, basename, dirname
from collections import defaultdict


matplotlib.use('Agg')

font = {
    'family': 'DejaVu Sans',
    'weight': 'normal',
    'size': 15
}


matplotlib.rc('font', **font)
matplotlib.rcParams["legend.columnspacing"] = .4


LINETYPES = ["-", "--", "-.", ":"]

PLOT_COLORS = [(.8, .6, 0), (.9, .2, .9), (.05, .4, .8), (0, .3, .3),
               (.6, .4, .4), (.2, .4, .1)]

FLAG_ASCII = platform.system() == 'Windows'


def pairwise(iterable):
    """pairwise returns an new iterable where each element is a pair of consecutive elements in the old iterable

    Args:
        iterable: Any iterable (list, tuple, etc.) 
            e.g., [1, 2, 3, 4, 5]

    Yields:
        Iterable: consecutive pairs of elements in iterable
            e.g., [(1, 2), (2, 3), (3, 4), (4, 5)]
    """
    it = iter(iterable)
    a = next(it, None)

    for b in it:
        yield (a, b)
        a = b


def assure_dir(path):
    if not exists(path):
        os.makedirs(path)


def reset_dir(path):
    if exists(path):
        shutil.rmtree(path, ignore_errors=True)

    if not exists(path):
        os.makedirs(path)


def remove_file_ext(filename):
    assert type(filename) is str
    pos = filename[::-1].find('.')
    return filename[:-pos-1] if pos != -1 else filename


def get_timer(time_start):
    """Stops the timer since time_start, and visualize the time differences in format.

    Args:
        time_start: a previous result of time.time()
    """
    seconds = time.time() - time_start
    assert(seconds >= 0)
    hours, seconds = divmod(seconds, 3600)
    minutes, seconds = divmod(seconds, 60)
    seconds, milsecs = divmod(seconds, 1)
    hours, minutes, seconds, milsecs = int(hours), int(
        minutes), int(seconds), int(milsecs * 1000)
    if hours > 0:
        time_str = '{:d}:{:02d}:{:02d}.{:03d}'.format(int(hours), int(minutes),
                                                      int(seconds), int(milsecs))
    elif minutes > 0:
        time_str = '{:d}:{:02d}.{:03d}'.format(
            int(minutes), int(seconds), int(milsecs))
    else:
        time_str = '{:02d}.{:03d}'.format(int(seconds), int(milsecs))

    print('Elapsed Time {:s}'.format(time_str))


def plot_advantage(data, name, ax=None, fontsize=15):
    """Plot the result of advantage values of different methods in simulations

    Args:
        data (dict): the result of simulation
        name (str): title name
        ax (optional): A preallocated subfig frame to plot in; New fig created if None. Defaults to None.
        fontsize (int, optional): font size of all text in the plot. Defaults to 15.

    Returns:
        None if ax is provided; Figure object otherwise.
    """
    data = defaultdict(lambda: None, data)

    random_advantage = data['ra']
    greedy_advantage = data['ga']
    perigee_advantage = data['pa']
    force_advantage = data['fa']
    monte_advantage = data['ma']
    nums_victims = data['vics']
    max_force_victims = data['fvic']
    tau = data['tau']
    ratio = data['ratio']
    epochs = data['epoch']

    ax_orig = ax
    if ax is None:
        fig = plt.figure(figsize=(7, 5))
        ax = fig.add_subplot(1, 1, 1)

    i = 0

    if random_advantage is not None:
        random_advantage = random_advantage.reshape((-1, random_advantage.shape[2]))

        ra_avg, ra_std = random_advantage.mean(0), random_advantage.std(0)

        color, linestyle = PLOT_COLORS[i], LINETYPES[i]

        ax.plot(nums_victims, ra_avg, color=color, linestyle=linestyle, alpha=.5, label='random')
        ax.fill_between(nums_victims, ra_avg-ra_std, ra_avg+ra_std, color=color, alpha=.1)

        i += 1

    if greedy_advantage is not None:
        ga_avg, ga_std = greedy_advantage.mean(0), greedy_advantage.std(0)

        color, linestyle = PLOT_COLORS[i], LINETYPES[i]

        ax.plot(nums_victims, ga_avg, color=color, linestyle=linestyle, alpha=.5, label='greedy')
        ax.fill_between(nums_victims, ga_avg-ga_std, ga_avg+ga_std, color=color, alpha=.1)

        i += 1

    if perigee_advantage is not None:
        pa_avg, pa_std = perigee_advantage.mean(0), perigee_advantage.std(0)
        color, linestyle = PLOT_COLORS[i], LINETYPES[i]

        for i, epoch in enumerate(epochs):
            ax.plot(nums_victims, pa_avg[:, i], color=color, linestyle=linestyle, alpha=.5, label='Peri')
            ax.fill_between(nums_victims, pa_avg[:, i]-pa_std[:, i], pa_avg[:, i]+pa_std[:, i], color=color, alpha=.1)

        i += 1

    if force_advantage is not None:
        color, linestyle = PLOT_COLORS[i], LINETYPES[i]
        fa_avg, fa_std = force_advantage.mean(0), force_advantage.std(0)
        ax.plot(np.arange(2, max_force_victims + 1), fa_avg, color=color, alpha=.5, label='max')
        ax.fill_between(np.arange(2, max_force_victims + 1), fa_avg-fa_std, fa_avg+fa_std, color=color, alpha=.1)
        i += 1

    if monte_advantage is not None:
        ma_avg, ma_std = monte_advantage.mean(0), monte_advantage.std(0)

        color, linestyle = PLOT_COLORS[i], LINETYPES[i]

        ax.plot(nums_victims, ma_avg, color=color, linestyle=linestyle, alpha=.5, label='monte')
        ax.fill_between(nums_victims, ma_avg-ma_std, ma_avg+ma_std, color=color, alpha=.1)
        i += 1

    ax.set_xlim(1, max(nums_victims) + 1)
    ax.set_title(name, zorder=1, fontsize=fontsize)

    if ax_orig is None:
        ax.set_xlabel('Number of adversarial links', fontsize=fontsize)
        ax.set_ylabel('Advantage (tau={})'.format(tau), fontsize=fontsize)
        ax.legend(fontsize=fontsize)
        plt.tight_layout()
        return fig
