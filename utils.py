__author__ = "Daan van Stigt"

import pickle
import os
import tempfile

import numpy as np


UD_LANG = {
    'cs': os.path.join('UD_Czech-PDT', 'cs_pdt'),
    'de': os.path.join('UD_German-GSD', 'de_gsd'),
    'en': os.path.join('UD_English-EWT', 'en_ewt'),
    'es': os.path.join('UD_Spanish-GSD', 'es_gsd'),
    'fr': os.path.join('UD_French-FTB', 'fr_ftb'),
    'hi': os.path.join('UD_Hindi-HDTB', 'hi_hdtb'),
    'nl': os.path.join('UD_Dutch-Alpino', 'nl_alpino'),
}


UD_SPLIT = {
    'train': '-ud-train.conllu',
    'dev': '-ud-dev.conllu',
    'test': '-ud-test.conllu'
}


def ceil_div(a, b):
    return ((a - 1) // b) + 1


def get_size(object):
    """Dump a pickle of object to get the size accurately."""
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, 'object.pkl')
    with open(path, 'wb') as f:
        pickle.dump(object, f)
    bytes = os.path.getsize(path)
    os.remove(path)
    gigabytes = bytes / 1e9
    return gigabytes


def softmax(x):
  """Numpy softmax function (normalizes rows)."""
  x -= np.max(x, axis=1, keepdims=True)
  x = np.exp(x)
  return x / np.sum(x, axis=1, keepdims=True)
