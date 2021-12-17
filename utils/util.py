import os
import pathlib
import random
from datetime import datetime

import numpy as np
import pandas as pd
import torch

horse = """               .,,.
             ,;;*;;;;,
            .-'``;-');;.
           /'  .-.  /*;;
         .'    \\d    \\;;               .;;;,
        / o      `    \\;    ,__.     ,;*;;;*;,
        \\__, _.__,'   \\_.-') __)--.;;;;;*;;;;,
         `""`;;;\\       /-')_) __)  `\' ';;;;;;
            ;*;;;        -') `)_)  |\\ |  ;;;;*;
            ;;;;|        `---`    O | | ;;*;;;
            *;*;\\|                 O  / ;;;;;*
           ;;;;;/|    .-------\\      / ;*;;;;;
          ;;;*;/ \\    |        '.   (`. ;;;*;;;
          ;;;;;'. ;   |          )   \\ | ;;;;;;
          ,;*;;;;\\/   |.        /   /` | ';;;*;
           ;;;;;;/    |/       /   /__/   ';;;
           '*jgs/     |       /    |      ;*;
                `""""`        `""""`     ;'"""  # Why jgs?

project_path = pathlib.Path(__file__).parent.parent


class Object(object):
    pass


def get_str_formatted_time() -> str:
    return datetime.now().strftime('%Y.%m.%d_%H.%M.%S.%f')


def ensure_dir(dirname):
    dirname = pathlib.Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)


def ensure_dirs(dirs):
    for dir_ in dirs:
        ensure_dir(dir_)


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
            ziph.write(os.path.join(root, file), os.path.relpath(os.path.join(root, file), os.path.join(path, '../..')))


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


def setup_torch_reproducibility(seed, convolution_determinism=True, convolution_benchmarking=False):
    """
    Seed torch, numpy and (python's) random with the given seed. If on GPU, determinism
    can be exchanged for speed by setting torch.backends.cudnn.deterministic to True
    and torch.backends.cudnn.benchmark to False, which is done by default parameters.
    For more information, visit https://pytorch.org/docs/stable/notes/randomness.html
    :param seed: seed to use
    :param convolution_determinism: value for torch.backends.cudnn.deterministic (True for reproducibility)
    :param convolution_benchmarking: value for torch.backends.cudnn.benchmark (False for reproducibility)
    :return: nothing to return
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = convolution_determinism
    torch.backends.cudnn.benchmark = convolution_benchmarking


def setup_torch_device(print_logs=True):
    device = torch.device("cuda") if torch.cuda.device_count() else torch.device("cpu")
    device_ids = list(range(torch.cuda.device_count()))

    if print_logs:
        print(torch.cuda.device_count())
        print(device_ids)
        print("Using device", device)

    return device
