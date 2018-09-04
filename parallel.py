import multiprocessing as mp

import numpy as np
from tqdm import tqdm

from features import shape, get_features
from utils import ceil_div

i = 0


def make_features_parallel(lines):
    def worker(lines, rank, return_dict):
        features = set()
        lines = tqdm(lines) if rank == 0 else lines
        for tokens in lines:
            for head in tokens:
                for dep in tokens:
                    features.update(get_features(head, dep, tokens))
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


def predict(token, tokens, weights, feature_dict):
    scores = []
    features = []
    for head in tokens:
        feats = get_features(head, token, tokens)
        score = score_fn(feats, weights, feature_dict)
        features.append(feats)
        scores.append(score)
    guess = np.argmax(scores)
    return guess, features


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


def worker(train_lines, rank, weights, feature_dict, dev_lines=None):
    train_lines = tqdm(train_lines) if rank == 0 else train_lines
    c = 0; n = 0
    for j, line in enumerate(train_lines):
        for token in line:
            guess, features = predict(token, line, weights, feature_dict)
            update(features[guess], features[token.head], weights, feature_dict)
            c += guess == token.head; n += 1
    if rank == 0:
        global i
        i += 1
        print(f'| Iter {i} | Correct guess {c:,}/{n:,} |')
        # print('Evaluating on dev set...')
        # if dev_lines is not None:
            # dev_acc = self.evaluate(dev_set[:20])
            # print(f'| Iter {i} | Train UAS {train_acc:.2f} | Dev UAS {dev_acc:.2f} |')
