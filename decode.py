from mst import get_best_graph
from eisner import eisner


class Decoder:
    def __init__(self, decoding='mst'):
        assert decoding in ('mst', 'eisner')
        self.decoding = decoding

    def __call__(self, scores):
        if self.decoding == 'mst':
            return get_best_graph(scores)
        else:
            return eisner(scores.T)
