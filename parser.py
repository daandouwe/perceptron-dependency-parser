__author__ = 'Daan van Stigt'

import multiprocessing as mp

import numpy as np

from perceptron import ArcPerceptron, LabPerceptron
from decode import Decoder
from features import get_features
from parallel import parse_parallel
from utils import softmax


class DependencyParser:
    def __init__(self, feature_opts={}, decoding='mst'):
        self.feature_opts = feature_opts
        self.arc_perceptron = ArcPerceptron()
        self.decoder = Decoder(decoding)

    def make_features(self, lines):
        self.arc_perceptron.make_features(lines)

    def train(self, niters, train_set, dev_set, approx=100, structured=None):
        # TODO: no structured training yet.
        # Train arc perceptron first.
        for i in range(1, niters+1):
            # Train arc perceptron for one epoch.
            c, n = self.arc_perceptron.train(niters, train_set)
            # Evaluate arc perceptron.
            train_acc, dev_acc = self.evaluate(train_set[:approx]), self.evaluate(dev_set[:approx])
            print(f'| Iter {i} | Correct guess {c:,}/{n:,} ~ {c/n:.2f} | Train UAS {train_acc:.2f} | Dev UAS {dev_acc:.2f} |')
            np.random.shuffle(train_set)

    def train_parallel(self, niters, lines, dev_set, nprocs=-1, approx=100):
        """Asynchronous lock-free (`Hogwild`) training of perceptron."""
        size = mp.cpu_count() if nprocs == -1 else nprocs
        print(f'Hogwild training with {size} processes...')
        self.arc_perceptron.prepare_for_parallel()
        for i in range(1, niters+1):
            # Train arc perceptron for one epoch in parallel.
            self.arc_perceptron.train_parallel(niters, lines, size)
            # Evaluate arc perceptron.
            train_acc = self.evaluate_parallel(lines[:approx])
            dev_acc = self.evaluate_parallel(dev_set[:approx])
            print(f'| Iter {i} | Train UAS {train_acc:.2f} | Dev UAS {dev_acc:.2f} |')
            np.random.shuffle(lines)
        self.arc_perceptron.restore_from_parallel()

    def parse(self, tokens):
        score_matrix = np.zeros((len(tokens), len(tokens)))
        all_features = dict()
        for i, dep in enumerate(tokens):
            all_features[i] = dict()
            for j, head in enumerate(tokens):
                features = get_features(head, dep, tokens, **self.feature_opts)
                score = self.arc_perceptron.score(features)
                score_matrix[i][j] = score
                all_features[i][j] = features
        probs = softmax(score_matrix)
        heads = self.decoder(probs)
        return heads, probs, all_features

    def accuracy(self, pred, gold):
        return 100 * sum((p == g for p, g in zip(pred, gold))) / len(pred)

    def evaluate(self, lines):
        acc = 0.0
        for line in lines:
            pred_heads, _, _ = self.parse(line)
            gold_heads = [token.head for token in line]
            acc += self.accuracy(pred_heads, gold_heads)
        return acc / len(lines)

    def evaluate_parallel(self, lines):
        acc = 0.0
        for line in lines:
            pred, _, _ = parse_parallel(
                line,
                self.arc_perceptron.weights,
                self.arc_perceptron.feature_dict,
                self.feature_opts,
                self.decoder)
            gold = [token.head for token in line]
            acc += self.accuracy(pred, gold)
        return acc / len(lines)

    def restore_from_parallel(self):
        self.arc_perceptron.restore_from_parallel()

    def average_weights(self):
        self.arc_perceptron.average_weights()

    def prune(self, eps):
        return self.arc_perceptron.prune(eps)

    def save(self, path):
        self.arc_perceptron.save(path)

    def load(self, path, training=False):
        self.arc_perceptron.load(path, training)

    # --------------------------------------- #
    # Temorpary attributes to ease transition #
    @property
    def weights(self):
        return self.arc_perceptron.weights

    @property
    def top_features(self):
        return self.arc_perceptron.top_features
