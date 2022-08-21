import numpy as np
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
from modules.basic import Opinion


def plot_opinion_shift(axis: plt.axes,
                       x_start: Opinion, x_end: Opinion,
                       color: str = 'blue', show: bool = True,
                       save: bool = False,
                       save_name: str = 'opinion_shift') -> None:
    if axis is None:
        _, ax = plt.subplots(nrows=1, ncols=1)
    axis.scatter(x_start, x_end, 1)  # ideologies
    axis.plot([0, 1], [0, 1], 'r--', transform=axis.transAxes)  # identity
    add_hist(axis, x_start, x_end, color=color)
    plt.xlabel('Initial Opinion')
    plt.ylabel('Final Opinion')
    plt.show(show)
    if save:
        save_name, _ = os.path.splitext(save_name)
        plt.savefig(save_name + '.png')


def add_hist(ax, x_start, x_end, color='blue') -> None:
    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("top", 1, pad=0.15, sharex=ax)
    axHisty = divider.append_axes("right", 1, pad=0.2, sharey=ax)
    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)
    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = max(np.max(np.abs(x_start)), np.max(np.abs(x_end)))
    lim = (int(xymax/binwidth) + 1)*binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x_start, density=True, bins=bins, color=color)
    # axHistx.set_title(legend)
    axHisty.hist(x_end, density=True, bins=bins, orientation='horizontal', color=color)
