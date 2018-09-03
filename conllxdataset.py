"""
Source: https://github.com/bastings/parser.
Edited slightly.
"""


import io
import os

from utils import XToken

ROOT_TOKEN = '<root>'
ROOT_TAG = 'ROOT'
ROOT_LABEL = '-root-'


def empty_conllx_example_dict():
  ex = {
    'id':      [],
    'form':    [],
    'lemma':   [],
    'cpos':    [],
    'pos':     [],
    'feats':   [],
    'head':    [],
    'deprel':  [],
    'phead':   [],
    'pdeprel': []
  }
  return ex


def start_conllx_example_dict():
  ex = {
    'id':      [0],
    'form':    [ROOT_TOKEN],
    'lemma':   ['_'],
    'cpos':    [ROOT_TAG],
    'pos':     [ROOT_TAG],
    'feats':   ['_'],
    'head':    [0],
    'deprel':  [ROOT_LABEL],
    'phead':   ['_'],
    'pdeprel': ['_']
  }
  return ex


def conllx_reader(f):
  """
  Return examples as a dictionary.
  Args:
    f:

  Returns:

  """

  ex = start_conllx_example_dict()

  for line in f:
    line = line.strip()

    if not line:
      yield ex
      ex = start_conllx_example_dict()
      continue

    parts = line.split()
    assert len(parts) == 10, "invalid conllx line: %s" % line

    _id, _form, _lemma, _cpos, _pos, _feats, _head, _deprel, _phead, _pdeprel = parts
    ex['id'].append(_id)
    ex['form'].append(_form)
    ex['lemma'].append(_lemma)
    ex['cpos'].append(_cpos)
    ex['pos'].append(_pos)
    ex['feats'].append(_feats)
    ex['head'].append(_head)
    ex['deprel'].append(_deprel)
    ex['phead'].append(_phead)
    ex['pdeprel'].append(_pdeprel)

  # possible last sentence without newline after
  if len(ex['form']) > 0:
    yield ex


class ConllXDataset:
  """Defines a CONLL-X Dataset. """
  def __init__(self, path):
    """Create a ConllXDataset given a path and field list.
    Arguments:
        path (str): Path to the data file.
        fields (dict[str: tuple(str, Field)]):
            The keys should be a subset of the columns, and the
            values should be tuples of (name, field).
            Keys not present in the input dictionary are ignored.
    """
    with io.open(os.path.expanduser(path), encoding="utf8") as f:
      self.examples = [d for d in conllx_reader(f)]

    self.tokens = [
            [XToken(*parts) for parts in zip(*d.values())]
        for d in self.examples]
