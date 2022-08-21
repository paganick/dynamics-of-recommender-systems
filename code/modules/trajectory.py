import matplotlib.pyplot as plt
import numpy as np
from abc import ABC
from typing import List
from modules.saveUtils import save_dict_to_file


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

    def get_item(self, key: str) -> np.ndarray:
        return self._trajectory[key]

    def get_number_entries_item(self, key: str) -> int:
        return self.get_item(key=key).shape[0]

    def add_item(self, key: str, item) -> None:
        item = np.asarray(item). reshape((1, -1))
        if self.is_empty(self.get_item(key=key)):
            self._trajectory[key] = item
        else:
            assert(self.get_item(key=key).shape[1] == item.shape[1]), 'The size must coincide.'
            self._trajectory[key] = np.concatenate((self.get_item(key), item), axis=0)

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

    def plot(self,
             keys: str or List[str],
             axis: plt.axes or None = None,
             color: str = 'blue',
             show: bool = True,
             t_start: int = 0,
             t_end: int = -1,
             save: bool = False,
             save_name: str = 'sim') -> None:
        if isinstance(keys, str):
            keys = [keys]
        if axis is None:  # TODO: fix this
            _, axis = plt.subplots(len(keys), 1)
        else:
            if len(axis) != len(keys):
                raise ValueError('The number of keys should coincide with the number of subplots.')
        x = np.arange(t_start, t_end)
        for i, k in enumerate(keys):
            if t_end == -1:
                x = np.arange(t_start, self.get_number_entries_item(key=k))
            axis[i].plot(x, self.get_item(key=k), c=color)
            axis[i].set_ylabel(k)
        if show:
            axis.show()
        if save:
            axis.savefig(save_name + '.png')

    def trajectory(self) -> dict:
        return self._trajectory

    def save_to_file(self, name: str, folder: str or None = None) -> None:
        save_dict_to_file(data=self.trajectory(), name=name, folder=folder)
