import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable


def add_hist(ax, x, y, c='blue', legend=''):
    divider = make_axes_locatable(ax)
    axHistx = divider.append_axes("top", 1, pad=0.15, sharex=ax)
    axHisty = divider.append_axes("right", 1, pad=0.2, sharey=ax)
    # make some labels invisible
    axHistx.xaxis.set_tick_params(labelbottom=False)
    axHisty.yaxis.set_tick_params(labelleft=False)
    # now determine nice limits by hand:
    binwidth = 0.1
    xymax = max(np.max(np.abs(x)), np.max(np.abs(y)))
    lim = (int(xymax/binwidth) + 1)*binwidth

    bins = np.arange(-lim, lim + binwidth, binwidth)
    axHistx.hist(x, density=True, bins=bins, color=c)
    # axHistx.set_title(legend)
    axHisty.hist(y, density=True, bins=bins, orientation='horizontal', color=c)
