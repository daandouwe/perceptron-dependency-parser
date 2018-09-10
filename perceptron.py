__author__ = 'Daan van Stigt'

import json
import multiprocessing as mp
from ctypes import c_float

import numpy as np
from tqdm import tqdm

from features import get_features
from decode import Decoder
from parallel import worker, make_features_parallel, parse_parallel
from tokens import Token
from utils import ceil_div, softmax


class Perceptron:
    """Base methods for a perceptron."""

    def __init__(self):
        pass

    def initialize_weights(self, feature_set):
        """Initialize weights with zero."""
        self.weights = dict((f, 0) for f in feature_set)
        self._totals = dict((f, 0) for f in feature_set)
        self._timestamps = dict((f, 0) for f in feature_set)

    def average_weights(self):
        """Average weights over all updates."""
        assert self.i > 0, 'cannot average weight'
        self.weights = dict((f, val/self.i) for f, val in self._totals.items())
        del self._totals, self._timestamps

    def prune(self, eps=1e-1):
        """Prune all features with weight smaller than eps."""
        before = len(self.weights)
        zeros = sum(1 for val in self.weights.values() if val == 0.0)
        self.weights = dict((f, val) for f, val in self.weights.items() if abs(val) > eps)
        return before, zeros

    def sort(self):
        """Sort features by weight."""
        self.weights = dict(sorted(self.weights.items(), reverse=True, key=lambda x: x[1]))

    def top_features(self, n):
        """Return the n features with the largest weight."""
        self.sort()
        return list(self.weights.items())[:n]

    def save(self, path, accuracy=None):
        """Save model features and weights in json format."""
        path = path + '.json' if not path.endswith('.json') else path
        self.sort()
        model = {
            'accuracy': accuracy,
            'feature_opts': self.feature_opts,
            'weights': self.weights
        }
        with open(path, 'w') as f:
            json.dump(model, f, indent=4)

    def load(self, path, training=False):
        """Load model features and weights from json format."""
        path = path + '.json' if not path.endswith('.json') else path
        with open(path, 'r') as f:
            model = json.load(f)
        weights, feature_opts, accuracy = model['weights'], model['feature_opts'], model['accuracy']
        self.weights = weights
        self.feature_opts = feature_opts
        if training:
            # If we wish to continue training we will need these.
            self._totals = dict((f, 0) for f in weights.keys())
            self._timestamps = dict((f, 0) for f in weights.keys())
        return accuracy, feature_opts


class ArcPerceptron(Perceptron):
    """Perceptron to score dependency arcs."""

    def __init__(self, feature_opts=dict()):
        self.feature_opts = feature_opts
        self.i = 0

    def make_features(self, lines, parallel=False):
        """Create the feature-set from gold head-dep pairs in the lines."""
        assert isinstance(lines, list)
        assert all(isinstance(line, list) for line in lines)
        assert all(isinstance(token, Token) for line in lines for token in line)
        if not parallel:
            features = set()
            for tokens in tqdm(lines):
                for dep in tokens:
                    head = tokens[dep.head]
                    features.update(get_features(head, dep, tokens, **self.feature_opts))
        else:
            features = make_features_parallel(lines, self.feature_opts)
        self.initialize_weights(features)
        del features

    def make_all_features(self, lines, parallel=False):
        """Create the feature-set from *all* head-dep pairs in the lines.

        We need the features of _all_ possible head-dep combinations
        in order to produce full score matrices at prediction time.
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

    def train(self, niters, train_set):
        c = 0; n = 0
        for j, line in enumerate(tqdm(train_set)):
            for token in line:
                guess, features = self.predict(token, line)
                self.update(features[guess], features[token.head])
                c += guess == token.head; n += 1
        return c, n

    def prepare_for_parallel(self):
        """Prepare shared memory for parallel training."""
        self.feature_dict = dict((f, i) for i, f in enumerate(self.weights.keys()))
        self.weights = mp.Array(
            c_float,
            np.zeros(len(self.feature_dict)),
            lock=False)  # Hogwild!

    def restore_from_parallel(self):
        """Restore weights dictionary from shared memory after parallel training."""
        # Restore weights.
        self.weights = dict((f, self.weights[i]) for f, i in self.feature_dict.items())
        del self.feature_dict

    def train_parallel(self, niters, lines, nprocs):
        chunk_size = ceil_div(len(lines), nprocs)
        partitioned = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
        processes = []
        for rank in range(nprocs):
            p = mp.Process(
                target=worker,
                args=(partitioned[rank], rank, self.weights, self.feature_dict, self.feature_opts))
            p.start()
            processes.append(p)
        for p in processes:
            p.join()


class LabPerceptron(Perceptron):
    """Perceptron to score labels for dependency arcs."""
    pass



# class Perceptron:
#     def __init__(self, decoding='mst', feature_opts={}):
#         self.decoder = Decoder(decoding)
#         self.feature_opts = feature_opts
#         self.i = 0
#
#     def initialize_weights(self, feature_set):
#         self.weights = dict((f, 0) for f in feature_set)
#         self._totals = dict((f, 0) for f in feature_set)
#         self._timestamps = dict((f, 0) for f in feature_set)
#
#     def make_features(self, lines, parallel=False):
#         """Create the feature-set from all head-dep pairs in the lines.
#
#         We need the features of _all_ possible head-dep combinations
#         in order to produce full score matrices at prediction time.
#         Note: this can take some time if `lines` is long...
#         """
#         assert isinstance(lines, list)
#         assert all(isinstance(line, list) for line in lines)
#         assert all(isinstance(token, Token) for line in lines for token in line)
#         if not parallel:
#             features = set()
#             for tokens in tqdm(lines):
#                 for head in tokens:
#                     for dep in tokens:
#                         features.update(get_features(head, dep, tokens, **self.feature_opts))
#         else:
#             features = make_features_parallel(lines, self.feature_opts)
#         self.initialize_weights(features)
#         del features
#
#     def score(self, features):
#         score = 0.0
#         for f in features:
#             if f not in self.weights:
#                 continue
#             score += self.weights[f]
#         return score
#
#     def predict(self, token, tokens, weights=None):
#         """Gready head prediction used for training."""
#         scores = []
#         features = []
#         for head in tokens:
#             feats = get_features(head, token, tokens, **self.feature_opts)
#             score = self.score(feats)
#             features.append(feats)
#             scores.append(score)
#         guess = np.argmax(scores)
#         return guess, features
#
#     def update(self, guess_features, true_features):
#         def upd_feat(f, v):
#             if f not in self.weights:
#                 pass
#             else:
#                  # For the number of parameter updates in which we skipped this feature.
#                 nr_iters_at_this_weight = self.i - self._timestamps[f]
#                 self._totals[f] += nr_iters_at_this_weight * self.weights[f]
#                 self.weights[f] += v
#                 self._timestamps[f] = self.i
#
#         self.i += 1
#         for f in true_features:
#             upd_feat(f, 1.0)
#         for f in guess_features:
#             upd_feat(f, -1.0)
#
#     def train_step(self, line, structured=False):
#         def structured_update(line):
#             """Heads are predicted with tree algorithm."""
#             c = 0
#             heads, _, all_features = self.parse(line)
#             for i in range(len(line)):
#                 token = line[i]
#                 guess = heads[i]
#                 self.update(all_features[i][guess], all_features[i][token.head])
#                 c += guess == token.head
#             return c
#
#         def greedy_update(line):
#             """Heads are predicted greedily per dependant."""
#             c = 0
#             for token in line:
#                 guess, features = self.predict(token, line)
#                 self.update(features[guess], features[token.head])
#                 c += guess == token.head
#             return c
#
#         if structured:
#             c = structured_update(line)
#         else:
#             c = greedy_update(line)
#         return c
#
#     def train(self, niters, lines, dev_set, structured=False, approx=100):
#         for i in range(1, niters+1):
#             c = 0; n = 0
#             for j, line in enumerate(tqdm(lines)):
#                 correct = self.train_step(line, structured=structured)
#                 c += correct; n += len(line)
#             train_acc, dev_acc = self.evaluate(lines[:approx]), self.evaluate(dev_set[:approx])
#             print(f'| Iter {i} | Correct guess {c:,}/{n:,} ~ {c/n:.2f} | Train UAS {train_acc:.2f} | Dev UAS {dev_acc:.2f} |')
#             np.random.shuffle(lines)
#
#     def train_parallel(self, niters, lines, dev_set, nprocs=-1, approx=100):
#         """Asynchronous lock-free (`Hogwild`) training of perceptron."""
#         size = mp.cpu_count() if nprocs == -1 else nprocs
#         print(f'Hogwild training with {size} processes...')
#         chunk_size = ceil_div(len(lines), size)
#         # Make a shared array of the weights (cannot make a shared dict),
#         # but will restore this to dict after training.
#         self.feature_dict = dict((f, i) for i, f in enumerate(self.weights.keys()))
#         self.weights = mp.Array(
#             c_float,
#             np.zeros(len(self.feature_dict)),
#             lock=False  # Hogwild!
#         )
#         for i in range(1, niters+1):
#             partitioned = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
#             processes = []
#             for rank in range(size):
#                 p = mp.Process(
#                     target=worker,
#                     args=(partitioned[rank], rank, self.weights, self.feature_dict, self.feature_opts))
#                 p.start()
#                 processes.append(p)
#             for p in processes:
#                 p.join()
#             # TODO: make this less hacky:
#             train_acc = self.evaluate_parallel(
#                 lines[:approx], self.weights, self.feature_dict, self.feature_opts, self.decoder)
#             dev_acc = self.evaluate_parallel(
#                 dev_set[:approx], self.weights, self.feature_dict, self.feature_opts, self.decoder)
#             print(f'| Iter {i} | Train UAS {train_acc:.2f} | Dev UAS {dev_acc:.2f} |')
#             np.random.shuffle(lines)
#         # Restore weights.
#         self.weights = dict((f, self.weights[i]) for f, i in self.feature_dict.items())
#         del self.feature_dict
#
#     def average_weights(self):
#         self.weights = dict((f, val/self.i) for f, val in self._totals.items())
#         del self._totals, self._timestamps
#
#     def accuracy(self, pred, gold):
#         return 100 * sum((p == g for p, g in zip(pred, gold))) / len(pred)
#
#     def evaluate(self, lines):
#         acc = 0.0
#         for line in lines:
#             pred, _, _ = self.parse(line)
#             gold = [token.head for token in line]
#             acc += self.accuracy(pred, gold)
#         return acc / len(lines)
#
#     def evaluate_parallel(self, lines, weights, feature_dict, feature_opts, decoder):
#         acc = 0.0
#         for line in lines:
#             pred, _, _ = parse_parallel(line, weights, feature_dict, feature_opts, decoder)
#             gold = [token.head for token in line]
#             acc += self.accuracy(pred, gold)
#         return acc / len(lines)
#
#     def parse(self, tokens):
#         score_matrix = np.zeros((len(tokens), len(tokens)))
#         all_features = dict()
#         for i, dep in enumerate(tokens):
#             all_features[i] = dict()
#             for j, head in enumerate(tokens):
#                 features = get_features(head, dep, tokens, **self.feature_opts)
#                 score = self.score(features)
#                 score_matrix[i][j] = score
#                 all_features[i][j] = features
#         probs = softmax(score_matrix)
#         heads = self.decoder(probs)
#         return heads, probs, all_features
#
#     def prune(self, eps=1e-1):
#         before = len(self.weights)
#         zeros = sum(1 for val in self.weights.values() if val == 0.0)
#         self.weights = dict((f, val) for f, val in self.weights.items() if abs(val) > eps)
#         return before, zeros
#
#     def sort(self):
#         """Sort features by weight."""
#         self.weights = dict(sorted(self.weights.items(), reverse=True, key=lambda x: x[1]))
#
#     def save(self, path):
#         path = path + '.json' if not path.endswith('.json') else path
#         self.sort()
#         with open(path, 'w') as f:
#             json.dump(self.weights, f, indent=4)
#
#     def load(self, path, training=False):
#         path = path + '.json' if not path.endswith('.json') else path
#         with open(path, 'r') as f:
#             weights = json.load(f)
#         self.weights = weights
#         if training:
#             # If we wish to continue training we will need these.
#             self._totals = dict((f, 0) for f in weights.keys())
#             self._timestamps = dict((f, 0) for f in weights.keys())
#
#     def top_features(self, n):
#         self.sort()
#         return list(self.weights.items())[:n]
