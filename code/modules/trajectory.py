import matplotlib.pyplot as plt
import numpy as np
import os
from abc import ABC
from typing import List
from modules.saveUtils import save_dict_to_file, save_figure


class Trajectory(ABC):
    def __init__(self, keys: list or str) -> None:
        self._trajectory = {}
        if isinstance(keys, str):
            keys = [keys]
        for k in keys:
            self._trajectory[k] = self.empty_vector()

    @staticmethod
    def empty_vector() -> np.ndarray:
        return np.array([np.nan])

    @staticmethod
    def is_empty(x: np.ndarray) -> bool:
        return np.any(np.isnan(x))

    def __getitem__(self, item: str) -> np.ndarray:
        if isinstance(item, str):
            return self._trajectory[item]
        else:
            raise ValueError('Unknown input type.')

    def get_number_entries_item(self, key: str) -> int:
        return self.__getitem__(item=key).shape[0]

    def add_item(self, key: str, item) -> None:
        item = np.asarray(item). reshape((1, -1))
        if self.is_empty(self.__getitem__(item=key)):
            self._trajectory[key] = item
        else:
            assert(self.__getitem__(item=key).shape[1] == item.shape[1]), 'The size must coincide.'
            self._trajectory[key] = np.concatenate((self.__getitem__(key), item), axis=0)

    def append(self, keys: List[str] or str, items) -> None:
        if isinstance(keys, list):
            assert isinstance(items, list), 'items should be a list'
            assert len(items) == len(keys), 'items should have the same length as keys.'
            for i, k in enumerate(keys):
                self.add_item(key=k, item=items[i])
        elif isinstance(keys, str):
            self.add_item(key=keys, item=items)
        else:
            raise ValueError

    def reset(self) -> None:
        for key in self._trajectory:
            self._trajectory[key] = self.empty_vector()

    def plot(self,
             keys: str or List[str],
             axis: plt.axes or None = None,
             color: str = 'blue',
             show: bool = True,
             t_start: int = 0,
             t_end: int = -1,
             save: bool = False,
             folder: str = None,
             name: str = None) -> None:
        if isinstance(keys, str):
            keys = [keys]
        if axis is None:
            _, axis = plt.subplots(len(keys), 1)
        else:
            if len(axis) != len(keys):
                raise ValueError('The number of keys should coincide with the number of subplots.')
        x = np.arange(t_start, t_end)
        for i, k in enumerate(keys):
            if t_end == -1:
                x = np.arange(t_start, self.get_number_entries_item(key=k))
            axis[i].plot(x, self.__getitem__(item=k), c=color)
            axis[i].set_ylabel(k)
            axis[i].set_xlim(left=x[0], right=x[-1])
            if i <= len(keys) - 2:
                axis[i].xaxis.set_ticklabels([])
        if show:
            plt.show()
        if save:
            save_figure(name=name, folder=folder)

    def trajectory(self) -> dict:
        return self._trajectory

    def save_to_file(self, name: str, folder: str or None = None) -> None:
        save_dict_to_file(data=self.trajectory(), name=name, folder=folder)
