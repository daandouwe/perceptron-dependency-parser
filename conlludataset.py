import io
import os


def empty_conllu_example_dict():
  ex = {
    'id':      [],
    'form':    [],
    'lemma':   [],
    'pos':     [],
    'upos':    [],
    'feats':   [],
    'head':    [],
    'deprel':  [],
    'deps':    [],
    'misc':    [],
    'chars':   []
  }
  return ex


def conllu_reader(f):
  """
  Return examples as a dictionary.
  Args:
    f:

  Returns:

  """

  ex = empty_conllu_example_dict()

  for line in f:
    line = line.strip()

    if not line:
      yield ex
      ex = empty_conllu_example_dict()
      continue

    # comments
    if line[0] == "#":
      continue

    parts = line.split()
    assert len(parts) == 10, "invalid conllx line: %s" % line

    _id, _form, _lemma, _upos, _xpos, _feats, _head, _deprel, _deps, _misc = parts

    ex['id'].append(_id)
    ex['form'].append(_form)
    ex['lemma'].append(_lemma)
    ex['upos'].append(_upos)
    ex['pos'].append(_xpos)
    ex['feats'].append(_feats)

    # TODO: kan dit? (0 is root)
    if _head == "_":
      _head = 0

    ex['head'].append(_head)
    ex['deprel'].append(_deprel)
    ex['deps'].append(_deps)
    ex['misc'].append(_misc)
    ex['chars'].append(list(_form))

  # possible last sentence without newline after
  if len(ex['form']) > 0:
    yield ex


class ConllUDataset:
  """Defines a CONLL-U Dataset. """
  def __init__(self, path):
    """Create a ConllUDataset given a path and field list.
    Arguments:
        path (str): Path to the data file.
        fields (dict[str: tuple(str, Field)]):
            The keys should be a subset of the columns, and the
            values should be tuples of (name, field).
            Keys not present in the input dictionary are ignored.
    """

    with io.open(os.path.expanduser(path), encoding="utf8") as f:
      self.examples = [d for d in conllu_reader(f)]
