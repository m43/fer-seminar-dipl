import numpy as np
import os
import pandas as pd
import pathlib
import random
import torch
from datetime import datetime


project_path = pathlib.Path(__file__).parent.parent


class Object(object):
    pass


def get_str_formatted_time() -> str:
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S')


def ensure_dir(dirname):
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def ensure_dirs(dirs):
    for dir_ in dirs:
        ensure_dir(dir_)


def sigmoid(x, beta=1):
    return 1. / (1. + np.exp(-beta * x))


def softmax(X):
    X_stable = X - np.expand_dims(X.max(axis=-1), axis=-1)
    exp_X_stable = np.exp(X_stable)
    result = exp_X_stable / np.expand_dims(exp_X_stable.sum(axis=-1), axis=-1)
    return result


def relu(x):
    return np.maximum(0, x)


def zipdir(path, ziph):
    """
    Usage example:
    zipf = zipfile.ZipFile(results_path + ".zip", 'w', zipfile.ZIP_DEFLATED)
    zipdir(results_path, zipf)
    zipf.close()

    Source: https://stackoverflow.com/questions/41430417/using-zipfile-to-create-an-archive

    :param path: Path to dir to zip
    :param ziph: zipfile handle
    :return:
    """
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '..')))


class MetricTracker:
    def __init__(self, name, *keys):
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'])
        self.name = name
        self.reset()

    def get_name(self):
        return self.name

    def reset(self):
        for col in self._data.columns:
            self._data[col].values[:] = 0

    def update(self, key, value, n=1):
        self._data.total[key] += value * n
        self._data.counts[key] += n
        self._data.average[key] = self._data.total[key] / self._data.counts[key]

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)


def setup_torch_reproducibility(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup_torch_device(print_logs=True):
    device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
    device_ids = list(range(torch.cuda.device_count()))

    if print_logs:
        print(torch.cuda.device_count())
        print(device_ids)
        print("Using device", device)

    return device
