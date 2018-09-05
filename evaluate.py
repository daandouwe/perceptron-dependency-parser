__author__ = 'Daan van Stigt'

import os
import subprocess
import re
import pickle

from tqdm import tqdm

from parser import DependencyParser
from utils import UD_LANG, UD_SPLIT

def predict(model, lines):
    pred = []
    for i, line in enumerate(tqdm(lines)):
        final = i == len(lines) - 1
        pred_line = []
        pred_heads, _, _ = model.parse(line)
        for i, token in enumerate(line[1:], 1):  # Discard Root.
            token.head = pred_heads[i]
            pred_line.append(str(token))
        if not final:
            pred_line.append('')  # Empty line.
        pred.append('\n'.join(pred_line))
    return pred


def call_conllx_eval_script(gold_path, pred_path):
    s = subprocess.check_output(
        ['perl', 'scripts/eval.pl', '-g', gold_path, '-s', pred_path, '-q']
    ).decode('utf-8')
    scores = re.findall('([0-9]?[0-9]\.[0-9][0-9]) %', s)
    las, uas, lab_acc = [float(score) for score in scores]
    return las, uas, lab_acc


def call_conllu_eval_script(gold_path, pred_path):
    s = subprocess.check_output(
        ['scripts/conll18_ud_eval.py', '-v', gold_path, pred_path]
    ).decode('utf-8')
    return s


def evaluate(args):
    from main import get_data  # TODO: cannot import at top... Some circular dependency?

    print(f'Loading data from `{args.data}`...')
    _, dev_dataset, test_dataset = get_data(args)

    print(f'Loading model from `{args.model}`...')
    feature_opts = get_feature_opts(args.features)
    model = DependencyParser(args.decoder, feature_opts)
    model.load(args.model)

    print(f'Parsing development set...')
    dev_pred = predict(model, dev_dataset.tokens)
    print(f'Parsing test set...')
    test_pred = predict(model, test_dataset.tokens)

    ext = 'conll' if args.use_ptb else 'conllu'
    print(f'Writing out predictions in {ext} format to `{args.out}`...')
    dev_pred_path = os.path.join(args.out, f'dev.pred.{ext}')
    test_pred_path = os.path.join(args.out, f'test.pred.{ext}')
    with open(dev_pred_path, 'w') as f:
        print('\n'.join(dev_pred), file=f)
    with open(test_pred_path, 'w') as f:
        print('\n'.join(test_pred), file=f)

    print('Evaluating results...')
    data_dir = os.path.expanduser(args.data)
    if args.use_ptb:
        dev_gold_path = os.path.join(data_dir, f'dev.conll')
        test_gold_path = os.path.join(data_dir, f'test.conll')

        dev_las, dev_uas, dev_lab_acc = call_conllx_eval_script(dev_gold_path, dev_pred_path)
        test_las, test_uas, test_lab_acc = call_conllx_eval_script(test_gold_path, test_pred_path)

        # TODO: formatting can be done a little cleaner than this...
        dev_results = '\n'.join(
            (f'{"LAS":<10} {dev_las:3.2f}', f'{"UAS":<10} {dev_uas:3.2f}', f'{"Lab-acc":<10} {dev_lab_acc:3.2f}'))
        test_results = '\n'.join(
            (f'{"LAS":<10} {test_las:3.2f}', f'{"UAS":<10} {test_uas:3.2f}', f'{"Lab-acc":<10} {test_lab_acc:3.2f}'))
    else:
        data_path = os.path.join(data_dir, UD_LANG[args.lang])
        dev_gold_path = data_path + UD_SPLIT['dev']
        test_gold_path = data_path + UD_SPLIT['test']

        dev_results = call_conllu_eval_script(dev_gold_path, dev_pred_path)
        test_results = call_conllu_eval_script(test_gold_path, test_pred_path)

    # Print results to terminal.
    print('Development results:')
    print(dev_results)
    print()
    print('Test results:')
    print(dev_results)
    print()

    # Print results to file.
    dev_result_path = os.path.join(args.out, f'dev.{ext}.result')
    with open(dev_result_path, 'w') as f:
        print(f'Gold file: {dev_gold_path}', file=f)
        print(f'Predicted file: {dev_pred_path}', file=f)
        print(file=f)
        print(dev_results, file=f)

    test_result_path = os.path.join(args.out, f'test.{ext}.result')
    with open(dev_result_path, 'w') as f:
        print(f'Gold file: {dev_gold_path}', file=f)
        print(f'Predicted file: {dev_pred_path}', file=f)
        print(file=f)
        print(test_results, file=f)
