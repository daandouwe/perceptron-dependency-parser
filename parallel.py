__author__ = 'Daan van Stigt'

import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from features import shape, get_features
from utils import ceil_div, softmax


def make_features_parallel(lines, feature_opts):
    def worker(lines, rank, return_dict):
        features = set()
        lines = tqdm(lines) if rank == 0 else lines
        for tokens in lines:
            for head in tokens:
                for dep in tokens:
                    features.update(get_features(head, dep, tokens, **feature_opts))
        return_dict[rank] = features

    size = mp.cpu_count()
    print(f'Making feature set with {size} processes...')
    chunk_size = ceil_div(len(lines), size)
    partitioned = [lines[i:i+chunk_size] for i in range(0, len(lines), chunk_size)]
    manager = mp.Manager()
    return_dict = manager.dict()
    processes = []
    for rank in range(size):
        p = mp.Process(
            target=worker,
            args=(partitioned[rank], rank, return_dict))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
    print('Joining features...')
    features = set()
    for rank in tqdm(range(size)):
        features.update(return_dict[rank])
    return features


def score_fn(features, weights, feature_dict):
    score = 0.0
    for f in features:
        if f not in feature_dict:
            continue
        i = feature_dict[f]
        score += weights[i]  # weights is now an array.
    return score


def predict(token, tokens, weights, feature_dict, feature_opts):
    scores = []
    features = []
    for head in tokens:
        feats = get_features(head, token, tokens, **feature_opts)
        score = score_fn(feats, weights, feature_dict)
        features.append(feats)
        scores.append(score)
    guess = np.argmax(scores)
    return guess, features


def parse_parallel(tokens, weights, feature_dict, feature_opts, decoder):
    score_matrix = np.zeros((len(tokens), len(tokens)))
    all_features = dict()
    for i, dep in enumerate(tokens):
        all_features[i] = dict()
        for j, head in enumerate(tokens):
            features = get_features(head, dep, tokens, **feature_opts)
            score = score_fn(features, weights, feature_dict)
            score_matrix[i][j] = score
            all_features[i][j] = features
    probs = softmax(score_matrix)
    heads = decoder(probs)
    return heads, probs, all_features


def update(guess_features, true_features, weights, feature_dict):
    def upd_feat(f, v):
        if f not in feature_dict:
            pass
        else:
            # nr_iters_at_this_weight = self.i - self._timestamps[f]  # For the times we skipped this feature.
            # self._totals[f] += nr_iters_at_this_weight * weights[f]
            i = feature_dict[f]
            weights[i] += v
            # self._timestamps[f] = self.i

    # self.i += 1
    for f in true_features:
        upd_feat(f, 1.0)
    for f in guess_features:
        upd_feat(f, -1.0)


def worker(train_lines, rank, weights, feature_dict, feature_opts, dev_lines=None):
    train_lines = tqdm(train_lines) if rank == 0 else train_lines
    for j, line in enumerate(train_lines):
        for token in line:
            guess, features = predict(token, line, weights, feature_dict, feature_opts)
            update(features[guess], features[token.head], weights, feature_dict)
