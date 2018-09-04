__author__ = "Daan van Stigt"

import pickle
import os
import tempfile


def ceil_div(a, b):
    return ((a - 1) // b) + 1


def get_size(object):
    """Dump a pickle and get object size."""
    tmp = tempfile.gettempdir()
    path = os.path.join(tmp, 'object.pkl')
    with open(path, 'wb') as f:
        pickle.dump(object, f)
    bytes = os.path.getsize(path)
    os.remove(path)
    gigabytes = bytes / 1e9
    return gigabytes
