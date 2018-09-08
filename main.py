#!/usr/bin/env python
__author__ = 'Daan van Stigt'

import argparse
import os
import subprocess
import re
import pickle

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from conllxdataset import ConllXDataset
from conlludataset import ConllUDataset
from features import get_feature_opts
from parser import DependencyParser
from evaluate import evaluate
from utils import get_size, UD_LANG, UD_SPLIT


def get_data(args):
    data_dir = os.path.expanduser(args.data)
    if args.use_ptb:
        train_dataset = ConllXDataset(os.path.join(data_dir, 'train.conll'))
        dev_dataset = ConllXDataset(os.path.join(data_dir, 'dev.conll'))
        test_dataset = ConllXDataset(os.path.join(data_dir, 'test.conll'))
    else:
        data_path = os.path.join(data_dir, UD_LANG[args.lang])
        train_dataset = ConllUDataset(data_path + UD_SPLIT['train'])
        dev_dataset = ConllUDataset(data_path + UD_SPLIT['dev'])
        test_dataset = ConllUDataset(data_path + UD_SPLIT['test'])
    return train_dataset, dev_dataset, test_dataset


def plot(args, n=5):
    print(f'Loading data from `{args.data}`...')
    _, dev_dataset, _ = get_data(args)
    print(f'Loading model from `{args.model}`...')
    feature_opts = get_feature_opts(args.features)
    model = DependencyParser(feature_opts, args.decoder)
    model.load(args.model)
    print(f'Saving plots of {n} score matrices at `image/`...')
    for i, tokens in enumerate(dev_dataset.tokens[:n]):
        _, probs, _ =  model.parse(tokens)
        plt.imshow(probs)
        plt.savefig(f'image/pred{i}.pdf')


def train(args):
    print(f'Loading dataset from `{args.data}`...')
    train_dataset, dev_dataset, test_dataset = get_data(args)
    train_tokens = train_dataset.tokens[:args.max_lines] # lose the last sentence with -1 but OK.
    dev_tokens = dev_dataset.tokens
    test_tokens = dev_dataset.tokens

    for dir in ('models', 'out'):
        if not os.path.exists(dir):
            os.makedirs(dir)

    # Make model.
    feature_opts = get_feature_opts(args.features)
    model = DependencyParser(feature_opts, args.decoder)
    if args.load:
        print(f'Loading model from-set from `{args.model}`...')
        model.load(args.model)
    else:
        print('Creating feature-set...')
        if len(feature_opts) > 0:
            print(f'Additional features: {", ".join(feature_opts.keys())}.')
        model.make_features(train_tokens)
    print(f'Number of features: {len(model.weights):,}.')
    print(f'Memory used by model: {get_size(model):.3f} GB.')


    # Train model.
    try:
        if args.parallel:
            model.train_parallel(args.epochs, train_tokens, dev_tokens, args.nprocs)
        else:
            objective = args.decoder if args.structured else 'greedy'
            print(f'Training with {objective} decoding...')
            model.train(args.epochs, train_tokens, dev_tokens, structured=args.structured)
    except KeyboardInterrupt:
        print('Exiting training early.')
        if args.parallel:
            model.restore_from_parallel()  # restores weights that were distributed

    # Evaluate model.
    print('Evaluating on dev set...')
    dev_acc = model.evaluate(dev_tokens)
    print(f'Dev UAS {dev_acc:.2f}')
    print('Top features:')
    top_features = model.top_features(30)
    print('\n'.join(f'{f} {v:.4f}' for f, v in top_features))
    print()

    if not args.parallel:
        print('Averaging model weights...')
        model.average_weights()

        # Evaluate again (to see difference).
        print('Evaluating on dev set...')
        dev_acc = model.evaluate(dev_tokens)
        print(f'Dev UAS {dev_acc:.2f}')
        print('Top features:')
        top_features = model.top_features(30)
        print('\n'.join(f'{f} {v:.4f}' for f, v in top_features))
        print()
    else:
        print('Weight averaging not yet supported for parallel.')

    before, zeros = model.prune(args.eps)
    print(f'Pruned weights with threshold {args.eps}...')
    print(f'Number of weights before: {before:,} ({zeros:,} exactly zero).')
    print(f'Number of pruned weights: {len(model.weights):,}.')

    print(f'Saving model to `{args.model}`...')
    model.save(args.model)

    print()
    print('Producing predictions in conll format and final evaluation...')
    evaluate(args)

    print('Finished.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train', 'eval', 'plot'],
                        help='choose action')
    parser.add_argument('--data', type=str, default='data/ud',
                        help='data dir')
    parser.add_argument('--lang', type=str, default='en',
                        choices=['cs', 'de', 'en', 'es', 'hi', 'fr', 'nl'],  # TODO: more langs
                        help='language (universal dependencies only)')
    parser.add_argument('--ptb-dir', type=str, default='~/data/stanford-ptb',
                        help='data dir for ptb')
    parser.add_argument('--use-ptb', action='store_true',
                        help='using penn treebank')
    parser.add_argument('--epochs', type=int, default=10,
                        help='epochs to train')
    parser.add_argument('--features', nargs='+', default=[],
                        help='space separated list of additional features',
                        choices=['dist', 'surround', 'between'])
    parser.add_argument('--parallel', action='store_true',
                        help='training in parallel')
    parser.add_argument('--nprocs', type=int, default=-1,
                        help='number of processors to use (default is all)')
    parser.add_argument('--structured', action='store_true',
                        help='using decoding algorithm to train on structured objective '
                        'specified by --decoder')
    parser.add_argument('--decoder', choices=['mst', 'eisner'], default='mst',
                        help='decoder used to extract tree from score matrix')
    parser.add_argument('--model', type=str, default='models/model.json',
                        help='path to save model to, or load model from')
    parser.add_argument('--out', type=str, default='out',
                        help='dir to put predicted conll files')
    parser.add_argument('--load', action='store_true',
                        help='load a pretrained model, specify which with --model')
    parser.add_argument('--eps', type=float, default=1e-2,
                        help='prune threshold')
    parser.add_argument('-n', '--max-lines', type=int, default=-1,
                        help='number of lines to train on.')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args)
    if args.mode == 'eval':
        evaluate(args)
    if args.mode == 'plot':
        plot(args)
