__author__ = 'Daan van Stigt'

from tokens import Token, XToken, UToken

START = '<sos>'
END = '<eos>'
START_POS = 'SOS'
END_POS = 'EOS'

START_XTOKEN = XToken(
    -1, START, START_POS, START_POS, '_', '_', -1, '_', '_', '_')

END_XTOKEN = XToken(
    -1, END, END_POS, END_POS, '_', '_', -1, '_', '_', '_')

START_UTOKEN = UToken(
    -1, START, START, START_POS, START_POS, '_', -1, START, '_', '_')

END_UTOKEN = UToken(
    -1, END, END, END_POS, END_POS, '_', -1, END, '_', '_')


def shape(word):
    """Inspired by spaCy's `token.shape` attribute."""
    punct = (',', '.', ';', ':', '?', '!', "'", '"')
    special = ('-', '/', '', '@', '#', '$', '%', '&')
    brackets = ('(', ')', '[', ']', '{', '}')
    shape = ''
    for char in word:
        if char.isupper():
            shape += 'X'
        elif char.islower():
            shape += 'x'
        elif char.isdigit():
            shape += 'd'
        elif char in brackets:
            shape += 'b'
        elif char in punct or char in special:
            shape += char
        else:
            shape += 'c'
    return shape


def get_features(head, dep, line, add_distance=False, add_surrounding=False, add_inbetween=False):
    """Feature-set loosely following McDonald et al. 2006."""
    assert isinstance(line, list)
    assert isinstance(head, Token), f'type {type(head)}'
    assert isinstance(dep, Token), f'type {type(dep)}'

    def get_token(line, id):
        assert isinstance(id, int), f'id not and int: {id}'
        type = 'utok' if isinstance(line[0], UToken) else 'xtok'
        if id in range(len(line)):
            token = line[id]
        elif id < 0:
            token = START_UTOKEN if type == 'utok' else START_XTOKEN
        else:
            token = END_UTOKEN if type == 'utok' else END_XTOKEN
        return token

    dep_min_2 = get_token(line, dep.id - 2)
    dep_min_1 = get_token(line, dep.id - 1)
    dep_plus_1 = get_token(line, dep.id + 1)
    dep_plus_2 = get_token(line, dep.id + 2)

    head_min_2 = get_token(line, head.id - 2)
    head_min_1 = get_token(line, head.id - 1)
    head_plus_1 = get_token(line, head.id + 1)
    head_plus_2 = get_token(line, head.id + 2)

    # Basic arc features
    features = (
        # Distance and position bias.
        'distance=%d' % (dep.id - head.id),
        'head dep id id=%d %d' % (head.id, dep.id),

        # Unigram features
        'head word=%s' % head.form,
        'head shape=%s' % shape(head.form),
        'head pos=%s' % head.pos,
        'head word pos=%s %s' % (head.form, head.pos),
        'head shape pos=%s %s' % (shape(head.form), head.pos),

        'dep word=%s' % dep.form,
        'dep shape=%s' % shape(dep.form),
        'dep pos=%s' % dep.pos,
        'dep word pos=%s %s' % (dep.form, dep.pos),
        'dep shape pos=%s %s' % (shape(dep.form), dep.pos),

        # Bigram (arc) features
        'head dep word word=%s %s' % (head.form, dep.form),
        'head dep pos pos=%s %s' % (head.pos, dep.pos),

        'head dep word pos=%s %s' % (head.form, dep.pos),
        'head dep pos word=%s %s' % (head.pos, dep.form),

        'head dep prefix prefix=%s %s' % (head.form[:2], dep.form[:2]),
        'head dep suffix suffix=%s %s' % (head.form[:-2], dep.form[:-2]),

        'head dep prefix suffix=%s %s' % (head.form[:2], dep.form[-2:]),
        'head dep suffix prefix=%s %s' % (head.form[-2:], dep.form[:2]),

        'head dep prefix prefix=%s %s' % (head.form[:3], dep.form[:3]),
        'head dep suffix suffix=%s %s' % (head.form[-3:], dep.form[-3:]),

        'head dep prefix suffix=%s %s' % (head.form[:3], dep.form[-3:]),
        'head dep suffix prefix=%s %s' % (head.form[-3:], dep.form[:3]),

        'head dep shape shape=%s %s' % (shape(head.form), shape(dep.form)),
        )

    if add_distance:
        # Stamp each of the basic features with the distance.
        features = tuple(f + ' (%d)' % (dep.id - head.id) for f in features[2:])  # distances do not need distance stamp

    if add_surrounding:
        features += (
            'head dep i i+1/i-1 i=%s %s/%s %s' % (head.pos, head_plus_1.pos, dep_min_1.pos, dep.pos),
            'head dep i-1 i/i-1 i=%s %s/%s %s' % (head_min_1.pos, head.pos, dep_min_1.pos, dep.pos),
            'head dep i i+1/i i+1=%s %s/%s %s' % (head.pos, head_plus_1.pos, dep.pos, dep_plus_1.pos),
            'head dep i-1 i/i i+1=%s %s/%s %s' % (head_min_1.pos, head.pos, dep.pos, dep_plus_1.pos)
        )

    if add_inbetween:
        betweens = line[head.id+1:dep.id] if head.id < dep.id else line[dep.id+1:head.id]
        features += tuple(
            ('head between dep=%s %s %s (%d %d)' % (
                head.pos, between.pos, dep.pos, between.id-head.id, dep.id-between.id)
                for between in betweens))

    return features


def get_feature_opts(features):
    opts = dict()
    for opt in features:
        if opt == 'dist':
            opts['add_distance'] = True
        if opt == 'surround':
            opts['add_surrounding'] = True
        if opt == 'between':
            opts['add_inbetween'] = True
    return opts
