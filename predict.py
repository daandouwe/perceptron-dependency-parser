__author__ = 'Daan van Stigt'

import os
import subprocess
import re
import pickle

import nltk
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from parser import DependencyParser
from features import get_feature_opts
from plot import plot_heatmap
from tokens import UToken, XToken
from conlludataset import ROOT_TOKEN, ROOT_TAG, ROOT_LABEL
from utils import UD_LANG, UD_SPLIT


JABBERWOCKY = [
    '’Twas brillig, and the slithy toves did gyre and gimble in the wabe.',
    'All mimsy were the borogoves, and the mome raths outgrabe.',
    'He took his vorpal sword in hand; long time the manxome foe he sought',
    'So rested he by the Tumtum tree and stood awhile in thought.',
]

EXAMPLES = [
    'One morning I shot an elephant in my pajamas.',
    'Time flies like an arrow; fruit flies like a banana.',
    'Buffalo buffalo Buffalo buffalo buffalo buffalo Buffalo buffalo.',
    "Will, will Will will Will Will's will?",
    'If police police police police, who police police police?',
    'The horse raced past the barn fell.'
]


def make_conll_tokens(tokens, tags):
    empty = '_'
    root = UToken(
        0, ROOT_TOKEN, ROOT_TOKEN, ROOT_TAG, ROOT_TAG, empty, 0, ROOT_LABEL, empty, empty)
    conll_tokens = [root]
    for i, (token, tag) in enumerate(zip(tokens, tags), 1):
        token = UToken(i, token, token, tag, tag, empty, -1, token, empty, empty)
        conll_tokens.append(token)
    return conll_tokens



def print_prediction(tokens, heads):
    for i, h in enumerate(heads[1:], 1):
        print(tokens[i].form + ' <-- ' + tokens[h].form)
    print()


def predict(args):
    tokenizer = nltk.tokenize.word_tokenize
    tagger = nltk.tag.perceptron.PerceptronTagger()

    print(f'Loading model from `{args.model}`...')
    feature_opts = get_feature_opts(args.features)
    model = DependencyParser(feature_opts, args.decoder)
    model.load(args.model)

    def predict_input(line):
        tokens = tokenizer(line)
        tagged = tagger.tag(tokens)
        print('> ' + ' '.join([f'{token}/{tag}' for token, tag in tagged]))
        tokens, tags = zip(*tagged)
        if args.no_tags:
            tags = len(tags) * ['_']
        tokens = make_conll_tokens(tokens, tags)
        heads, probs, _ = model.parse(tokens)
        return tokens, heads, probs

    if args.jabber:
        for i, line in enumerate(JABBERWOCKY, 1):
            tokens, heads, probs = predict_input(line)
            name = f'jab-{i}'
            if args.no_tags:
                name += '-notags'
            plot_heatmap([token.form for token in tokens], probs, name=name, ext='pdf')
            print_prediction(tokens, heads)
    if args.examples:
        for i, line in enumerate(EXAMPLES, 1):
            tokens, heads, probs = predict_input(line)
            name = f'ex-{i}'
            if args.no_tags:
                name += '-notags'
            plot_heatmap([token.form for token in tokens], probs, name=name, ext='pdf')
            print_prediction(tokens, heads)
    else:
        step = 0
        while True:
            step += 1
            line = input('Input: ')
            tokens, heads, probs = predict_input(line)
            name = f'{args.plot_name}-{step}'
            if args.no_tags:
                name += '-notags'
            plot_heatmap([token.form for token in tokens], probs, name=name, ext='pdf')
            print_prediction(tokens, heads)
