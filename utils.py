__author__ = "Daan van Stigt"

import pickle
import os
import tempfile


UD_LANG = {
    'en': os.path.join('UD_English-EWT', 'en_ewt'),
    'nl': os.path.join('UD_Dutch-Alpino', 'nl_alpino')
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
