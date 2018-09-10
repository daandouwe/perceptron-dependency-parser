import os

import numpy as np
import matplotlib.pyplot as plt

from parser import DependencyParser
from features import get_feature_opts


def plot_heatmap(tokens, probs, dir='image', name='input', ext='pdf'):

    # default_figsize = np.array(plt.rcParams.get('figure.figsize'))
    # figsize = 22 / len(tokens) * default_figsize
    longest_word = max(map(len, tokens))
    top_margin = max(longest_word * 0.2 / 9, 0.2)  # this setting seems to work well
    left_margin = max(longest_word * 0.2 / 12, 0.2)  # this setting seems to work well

    fig, ax = plt.subplots()
    heatmap = ax.pcolor(probs, cmap='viridis')

    ax.set_xticklabels(tokens, minor=False, rotation='vertical')
    ax.set_yticklabels(tokens, minor=False)

    ax.xaxis.tick_top()
    ax.set_xticks(np.arange(probs.shape[1]) + 0.5, minor=False)
    ax.set_yticks(np.arange(probs.shape[0]) + 0.5, minor=False)
    ax.invert_yaxis()

    plt.colorbar(heatmap)

    plt.subplots_adjust(left=left_margin, top=1-top_margin)

    name += '.' + ext
    path = os.path.join(dir, name)
    if ext == 'png':
        plt.savefig(path, dpi=300)
    else:
        plt.savefig(path)


def plot(args, n=5):
    from main import get_data  # Oops cyclical dependency

    print(f'Loading data from `{args.data}`...')
    _, dev_dataset, test_dataset = get_data(args)
    print(f'Loading model from `{args.model}`...')
    feature_opts = get_feature_opts(args.features)
    model = DependencyParser(feature_opts, args.decoder)
    model.load(args.model)
    print(f'Saving plots of {n} score matrices at `image/`...')
    for i, tokens in enumerate(test_dataset.tokens[:n]):
        _, probs, _ =  model.parse(tokens)
        plot_heatmap([token.form for token in tokens], probs, name=f'pred{i}', ext=args.ext)
