import os
import subprocess
import re
import pickle
from tqdm import tqdm


def predict(model, lines):
    pred = []
    for i, line in enumerate(tqdm(lines)):
        final = i == len(lines) - 1
        pred_line = []
        pred_heads, _ = model.parse(line)
        for i, token in enumerate(line[1:], 1):  # Discard Root.
            token.head = pred_heads[i]
            pred_line.append(str(token))
        if not final:
            pred_line.append('')  # Empty line.
        pred.append('\n'.join(pred_line))
    return pred


def call_eval_script(gold_path, pred_path):
    s = subprocess.check_output(
        ['perl', 'scripts/eval.pl', '-g', gold_path, '-s', pred_path, '-q']
    ).decode('utf-8')
    scores = re.findall('([0-9]?[0-9]\.[0-9][0-9]) %', s)
    las, uas, lab_acc = [float(score) for score in scores]
    return las, uas, lab_acc


def evaluate(args):
    from main import get_data  # TODO: cannot import at top... Some circular dependency?

    print(f'Loading data from `{args.data}`...')
    _, dev_dataset, test_dataset = get_data(args)

    print(f'Loading model from `{args.model}`...')
    with open(args.model, 'rb') as f:
        model = pickle.load(f)

    print(f'Parsing dev set...')
    dev_pred = predict(model, dev_dataset.tokens)
    print(f'Parsing test set...')
    test_pred = predict(model, test_dataset.tokens)

    # print(f'Writing out predictions in conll format to {args.out}...')
    ext = 'conllu' if args.ud else 'conll'
    dev_pred_path = os.path.join(args.out, f'dev.pred.{ext}')
    test_pred_path = os.path.join(args.out, f'test.pred.{ext}')
    with open(dev_pred_path, 'w') as f:
        print('\n'.join(dev_pred), file=f)
    with open(test_pred_path, 'w') as f:
        print('\n'.join(test_pred), file=f)

    print('Evaluating results...')
    # dev_result_path = os.path.join(args.out, f'dev.result.txt')
    # test_result_path = os.path.join(args.out, f'test.result.txt')
    data_path = os.path.expanduser(args.data)
    dev_gold_path = os.path.join(data_path, 'dev-stanford-raw.conll')  # TODO: remove name dependency here.
    test_gold_path = os.path.join(data_path, 'test-stanford-raw.conll')  # TODO: remove name dependency here.
    if args.ud:
        raise NotImplementedError('No UD evaluation availlable yet.')
    else:
        las, uas, lab_acc = call_eval_script(dev_gold_path, dev_pred_path)
        print(f'Dev LAS {las:5.2f} UAS {uas:5.2f} Lab-acc {lab_acc:5.2f}')
        las, uas, lab_acc = call_eval_script(test_gold_path, test_pred_path)
        print(f'Test LAS {las:5.2f} UAS {uas:5.2f} Lab-acc {lab_acc:5.2f}')
