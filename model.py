__author__ = "Daan van Stigt"

import pickle
import json
import multiprocessing as mp
from ctypes import c_float

import numpy as np
from tqdm import tqdm

from features import get_features
from decode import Decoder
from parallel import worker, make_features_parallel
from tokens import Token
from utils import ceil_div, softmax


class Perceptron:
    def __init__(self, decoding='mst', **kwargs):
        self.decoder = Decoder(decoding)
        self.feature_opts = kwargs
        self.i = 0

    def initialize_weights(self, feature_set):
        self.weights = dict((f, 0) for f in feature_set)
        self._totals = dict((f, 0) for f in feature_set)
        self._timestamps = dict((f, 0) for f in feature_set)

    def make_features(self, lines, parallel=False):
        """Create the feature-set from all head-dep pairs in the lines.

        We need the features of _all_ possible head-dep combinations
        in order to produce full score matrices at prediction time.
        Note: this can take some time if `lines` is long...
        """
        assert isinstance(lines, list)
        assert all(isinstance(line, list) for line in lines)
        assert all(isinstance(token, Token) for line in lines for token in line)
        if not parallel:
            features = set()
            for tokens in tqdm(lines):
                for head in tokens:
                    for dep in tokens:
                        features.update(get_features(head, dep, tokens, **self.feature_opts))
        else:
            features = make_features_parallel(lines, self.feature_opts)
        self.initialize_weights(features)
        del features

    def score(self, features):
        score = 0.0
        for f in features:
            if f not in self.weights:
                continue
            score += self.weights[f]
        return score

    def predict(self, token, tokens, weights=None):
        """Gready head prediction used for training."""
        scores = []
        features = []
        for head in tokens:
            feats = get_features(head, token, tokens, **self.feature_opts)
            score = self.score(feats)
            features.append(feats)
            scores.append(score)
        guess = np.argmax(scores)
        return guess, features

    def update(self, guess_features, true_features):
        def upd_feat(f, v):
            if f not in self.weights:
                pass
            else:
                 # For the number of parameter updates in which we skipped this feature.
                nr_iters_at_this_weight = self.i - self._timestamps[f]
                self._totals[f] += nr_iters_at_this_weight * self.weights[f]
                self.weights[f] += v
                self._timestamps[f] = self.i

        self.i += 1
        for f in true_features:
            upd_feat(f, 1.0)
        for f in guess_features:
            upd_feat(f, -1.0)

    def train(self, niters, lines, dev_set=None):
        """Training by greedy head selection."""
        for i in range(1, niters+1):
            c = 0; n = 0
            for j, line in enumerate(tqdm(lines)):
                for token in line:
                    guess, features = self.predict(token, line)
                    self.update(features[guess], features[token.head])
                    c += guess == token.head; n += 1
            train_acc = self.evaluate(lines[:100])  # a quick approximation
            message = f'| Iter {i} | Correct guess {c:,}/{n:,} | Train UAS {train_acc:.2f} |'
            if dev_set is not None:
                dev_acc = self.evaluate(dev_set[:100])  # a quick approximation
                message += f' Dev UAS {dev_acc:.2f} |'
            print(message)
            np.random.shuffle(lines)

    def train_struct(self, niters, lines, dev_set=None):
        """Training by structured head selection using tree-decoding."""
        for i in range(1, niters+1):
            c = 0; n = 0
            for j, line in enumerate(tqdm(lines)):
                heads, _, all_features = self.parse(line)
                for i in range(len(line)):
                    token = line[i]
                    guess = heads[i]
                    self.update(all_features[i][guess], all_features[i][token.head])
                    c += guess == token.head; n += 1
            train_acc = self.evaluate(lines[:100])  # a quick approximation
            message = f'| Iter {i} | Correct guess {c:,}/{n:,} | Train UAS {train_acc:.2f} |'
            if dev_set is not None:
                dev_acc = self.evaluate(dev_set[:100])  # a quick approximation
                message += f' Dev UAS {dev_acc:.2f} |'
            print(message)
            np.random.shuffle(lines)

    def train_parallel(self, niters, lines, dev_set=None):
        """Asynchronous lock-free (`Hogwild`) training of perceptron."""
        size = mp.cpu_count()
        chunk_size = ceil_div(len(lines), size)
        # Make a shared array of the weights (cannot make a shared dict).
        feature_dict = dict((f, i) for i, f in enumerate(self.weights.keys()))
        del self.weights  # free some memory space
        weights = mp.Array(
            c_float,
            np.zeros(len(feature_dict)),
            lock=False  # Hogwild!
        )
        print(f'Hogwild training with {size} processes...')
        for i in range(1, niters+1):
            partitioned = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
            processes = []
            for rank in range(size):
                p = mp.Process(
                    target=worker,
                    args=(partitioned[rank], rank, weights, feature_dict, self.feature_opts))
                p.start()
                processes.append(p)
            for p in processes:
                p.join()
            np.random.shuffle(lines)
        # Restore weights.
        self.weights = dict((f, weights[i]) for f, i in feature_dict.items())
        del weights, feature_dict

    def average_weights(self):
        print('Averaging model weights...')
        self.weights = dict((f, val/self.i) for f, val in self._totals.items())
        del self._totals, self._timestamps

    def accuracy(self, pred, gold):
        return 100 * sum((p == g for p, g in zip(pred, gold))) / len(pred)

    def evaluate(self, lines):
        acc = 0.0
        for line in lines:
            pred, _, _ = self.parse(line)
            gold = [token.head for token in line]
            acc += self.accuracy(pred, gold)
        return acc / len(lines)

    def parse(self, tokens):
        score_matrix = np.zeros((len(tokens), len(tokens)))
        all_features = dict()
        for i, dep in enumerate(tokens):
            all_features[i] = dict()
            for j, head in enumerate(tokens):
                features = get_features(head, dep, tokens, **self.feature_opts)
                score = self.score(features)
                score_matrix[i][j] = score
                all_features[i][j] = features
        probs = softmax(score_matrix)
        heads = self.decoder(probs)
        return heads, probs, all_features

    def prune(self, eps=1e-1):
        print(f'Pruning weights with threshold {eps}...')
        zeros = sum(1 for val in self.weights.values() if val == 0.0)
        print(f'Number of weights: {len(self.weights):,} ({zeros:,} exactly zero).')
        self.weights = dict((f, val) for f, val in self.weights.items() if abs(val) > eps)
        print(f'Number of pruned weights: {len(self.weights):,}.')

    def sort(self):
        """Sort features by weight."""
        self.weights = dict(sorted(self.weights.items(), reverse=True, key=lambda x: x[1]))

    def save(self, path):
        path = path + '.json' if not path.endswith('.json') else path
        self.sort()
        with open(path, 'w') as f:
            json.dump(self.weights, f, indent=4)

    def load(self, path, training=False):
        path = path + '.json' if not path.endswith('.json') else path
        with open(path, 'r') as f:
            weights = json.load(f)
        self.weights = weights
        if training:
            # If we wish to continue training we need these.
            self._totals = dict((f, 0) for f in weights.keys())
            self._timestamps = dict((f, 0) for f in weights.keys())

    def top_features(self, n):
        self.sort()
        return list(self.weights.items())[:n]
