import deepdish as dd
import matplotlib.pyplot as plt
import os


def save_dict_to_file(data: dict, name: str, folder: str or None = None) -> None:
    if not isinstance(data, dict):
        raise ValueError('Unknown input data.')
    if folder is None:
        folder = ''
    if not isinstance(name, str) or not isinstance(folder, str):
        raise ValueError('Unknown type of name or folder.')
    name, _ = os.path.splitext(name)
    if not os.path.isdir(folder) and folder != '':
        os.makedirs(folder)
    name, _ = os.path.splitext(name)
    if os.path.isfile(os.path.join(folder, name + '.h5')):
        path = os.path.join(folder, name + '_1.h5')
    else:
        path = os.path.join(folder, name + '.h5')
    dd.io.save(path, data)


def load_dict_from_data(name: str, folder: str or None = None) -> dict:
    if not isinstance(name, str) or not isinstance(folder, str):
        raise ValueError('Unknown type of name or folder.')
    name, _ = os.path.splitext(name)
    return dd.io.load(os.path.join(folder, name + '.h5'))


def save_figure(name: str = None, folder: str = None, extension: str = 'png') -> None:
    if name is None:
        raise ValueError('Input a name for the figure.')
    if folder is None:
        folder = '' # current folder
    if not os.path.isdir(folder) and folder != '':
        os.makedirs(folder)
    name, _ = os.path.splitext(name)  # remove extension
    plt.savefig(os.path.join(folder, name + '.png'), format=extension)
