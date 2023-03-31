import numpy as np
import matplotlib.pyplot as plt
from floweaver import *
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from typing import List, Tuple
from modules.basic import Opinion, Recommendation
from modules.saveUtils import save_figure


def plot_opinion_shift(axis: plt.axes,
                       axis_hist: tuple or None,
                       x_start: Opinion,
                       x_end: Opinion,
                       color: str = 'blue', show: bool = True,
                       save: bool = False,
                       name: str = None,
                       folder: str = None) -> None:
    if axis is None:
        _, axis = plt.subplots(nrows=1, ncols=1)
    if axis_hist is None:
        divider = make_axes_locatable(axis)
        axis_hist = (divider.append_axes("top", 1, pad=0.15, sharex=axis),
                     divider.append_axes("right", 1, pad=0.2, sharey=axis))
    # plot the ideology
    # axis.grid(visible=True)
    axis.scatter(x_start, x_end, 1, color=color)  # ideologies
    axis.plot([0, 1], [0, 1], 'r--', transform=axis.transAxes)  # identity
    axis.axis('equal')
    # plt.xlabel('Initial Opinion')
    # plt.ylabel('Final Opinion')
    # add histograms
    axis_hist[0].xaxis.set_tick_params(labelbottom=False)
    axis_hist[1].yaxis.set_tick_params(labelleft=False)
    bin_width = 0.1
    xy_max = max(np.max(np.abs(x_start)), np.max(np.abs(x_end)))
    lim = (int(xy_max / bin_width) + 1) * bin_width
    # plot
    bins = np.arange(-lim, lim + bin_width, bin_width)
    axis_hist[0].hist(x_start, density=True, bins=bins, color=color, alpha=0.7)
    axis_hist[1].hist(x_end, density=True, bins=bins, orientation='horizontal', color=color, alpha=0.7)
    if show:
        plt.show()
    if save:
        save_figure(name=name, folder=folder)


def plot_opinions_time(axis: plt.axes,
                       x: List[Opinion],
                       color: str = 'blue',
                       labels: List[str] = None,
                       show: bool = True,
                       save: bool = False,
                       name: str = None,
                       folder: str = None) -> None:
    if labels is None:
        raise ValueError('Please input labels.')
    if axis is None:
        _, axis = plt.subplots(nrows=1, ncols=1)
    assert len(labels) == len(x), 'The number of labels and data should coincide.'
    alpha = 0.7  # [0.1, 0.3, 0.5]
    colors = ['red', 'green', 'blue']
    for i, x_i in enumerate(x):
        axis.hist(x_i, bins=50, density=True, alpha=alpha, color=colors[i], label=labels[i])
    axis.legend()
    if show:
        plt.show()
    if save:
        save_figure(name=name, folder=folder)


def plot_sankey_single_population(x: Tuple[Opinion, Recommendation],
                                  color: str = 'blue',
                                  labels: List[str] = None,
                                  show: bool = True,
                                  save: bool = False,
                                  name: str = None,
                                  folder: str = None) -> tuple:
    # merge (by rounding)
    x = (np.round(x[0] * 2, 0) / 2).tolist(), (np.round(x[1] * 2, 0) / 2).tolist()
    # convert to panda
    data = pd.DataFrame(data={'source': x[0],
                              'target': x[1],
                              'value': [1] * len(x[0])})
    # make plot
    nodes = {
        'start': ProcessGroup(list(data['source'])),
        'end': ProcessGroup(list(data['target'])),
    }
    ordering = [['start'], ['end']]
    bundles = [Bundle('start', 'end')]
    d_start, d_end = list(data['source'].unique()), list(data['target'].unique())
    d_start.sort()
    d_end.sort()
    nodes['start'].partition = Partition.Simple('source', d_start)
    nodes['end'].partition = Partition.Simple('target', d_end)
    # compile plot and save it
    sdd = SankeyDefinition(nodes, bundles, ordering)
    weave(sdd, dataset=data).to_widget().auto_save_png('test.png')
    return sdd, data  # TODO: fix here


def plot_sankey_multiple_populations(x: List[Tuple[Opinion, Recommendation]],
                                     color: str = 'blue',
                                     labels: List[str] = None,
                                     show: bool = True,
                                     save: bool = False,
                                     name: str = None,
                                     folder: str = None) -> tuple:
    # merge (by rounding)
    data = []
    for i in range(len(x)):
        x[i] = (np.round(x[i][0] * 2, 0) / 2).tolist(), (np.round(x[i][1] * 2, 0) / 2).tolist()
        # convert to panda
        data.append(pd.DataFrame(data={'source': x[i][0],
                                       'target': x[i][1],
                                       'value': [1] * len(x[i][0]),
                                       'population': ['population_' + str(i)] * len(x[i][0])}))
    data = pd.concat(data)
    # make plot
    nodes = {'start': ProcessGroup(list(data['source'])),
             'end': ProcessGroup(list(data['target']))}
    ordering = [['start'], ['end']]
    bundles = [Bundle('start', 'end')]
    d_start, d_end = list(data['source'].unique()), list(data['target'].unique())
    d_start.sort()
    d_end.sort()
    nodes['start'].partition = Partition.Simple('source', d_start)
    nodes['end'].partition = Partition.Simple('target', d_end)
    # create population
    population = Partition.Simple('population', list(data['population'].unique()))
    # compile plot and save it
    sdd = SankeyDefinition(nodes, bundles, ordering, flow_partition=population)
    # colors
    weave(sdd, dataset=data, palette='Set1_3').to_widget().auto_save_png('test.png')
    return sdd, data  # TODO: fix here
