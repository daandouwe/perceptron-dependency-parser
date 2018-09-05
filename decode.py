__author__ = 'Daan van Stigt'

from mst import get_best_graph
from eisner import eisner


class Decoder:
    def __init__(self, decoding='mst'):
        assert decoding in ('mst', 'eisner')
        self.decoding = decoding

    def __call__(self, scores):
        if self.decoding == 'mst':
            return get_best_graph(scores)  # TODO: get a better name for this.
        else:
            return eisner(scores.T)  # TODO: get this consisten with mst: no transpose.

    def mst(self, scores):
        return get_best_graph(scores)

    def eisner(self, scores):
        return eisner(scores.T)
