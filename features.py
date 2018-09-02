from utils import XToken

START = '<sos>'
END = '<eos>'

START_TOKEN = XToken(
    -1, START, START, START, START, START, -1, START, START, START)

END_TOKEN = XToken(
    -1, END, END, END, END, END, -1, END, END, END)


def feature_dicts(
    dep,
    head,
    dep_min_2,
    dep_min_1,
    head_min_2,
    head_min_1,
    dep_plus_1,
    dep_plus_2,
    head_plus_1,
    head_plus_2
    ):
    head_features = {
        'self': [
            'head word={}'.format(head.form),
            'head suffix={}'.format(head.form[-3:]),
            'head pref={}'.format(head.form[:3]),
            'head shape={}'.format(shape(head.form)),
            'head pos={}'.format(head.pos),
            'head id={}'.format(head.id),
        ],
        'left': [
            'head-2 word={}'.format(head_min_2.form),
            'head-2 pos={}'.format(head_min_2.pos),
            'head-1 word={}'.format(head_min_1.form),
            'head-1 pos={}'.format(head_min_1.pos),
            'head-2 head-1 pos={}'.format(head_min_2.pos, head_min_1.pos),
            'head-2 head-1 word={}'.format(head_min_2.form, head_min_1.form),
        ],
        'right': [
            'head+1 word={}'.format(head_plus_1.form),
            'head+1 pos={}'.format(head_plus_1.pos),
            'head+2 word={}'.format(head_plus_2.form),
            'head+2 pos={}'.format(head_plus_2.pos),
            'head+1 head+2 word={} {}'.format(head_plus_1.form, head_plus_2.form),
            'head+1 head+2 pos={} {}'.format(head_plus_1.pos, head_plus_2.pos),
        ],
        'full': [
            'head context pos={} {} {} {}'.format(
                head_min_2.pos, head_min_1.pos, head_plus_2.pos, head_plus_1.pos),
        ]
    }
    dep_features = {
        'self': [
            'dep word={}'.format(dep.form),
            'dep suffix={}'.format(dep.form[-3:]),
            'dep pref={}'.format(dep.form[:3]),
            'dep shape={}'.format(shape(dep.form)),
            'dep pos={}'.format(dep.pos),
            'dep id={}'.format(dep.id),
        ],
        'left': [
            'dep-2 word={}'.format(dep_min_2.form),
            'dep-2 pos={}'.format(dep_min_2.pos),
            'dep-1 word={}'.format(dep_min_1.form),
            'dep-1 pos={}'.format(dep_min_1.pos),
            'dep-2 dep-1 pos={}'.format(dep_min_2.pos, dep_min_1.pos),
            'dep-2 dep-1 word={}'.format(dep_min_2.form, dep_min_1.form),
        ],
        'right': [
            'dep+1 word={}'.format(dep_plus_1.form),
            'dep+1 pos={}'.format(dep_plus_1.pos),
            'dep+2 word={}'.format(dep_plus_2.form),
            'dep+2 pos={}'.format(dep_plus_2.pos),
            'dep+1 dep+2 word={} {}'.format(dep_plus_1.form, dep_plus_2.form),
            'dep+1 dep+2 pos={} {}'.format(dep_plus_1.pos, dep_plus_2.pos),
        ],
        'full': [
            'dep context pos={} {} {} {}'.format(
                dep_min_2.pos, dep_min_1.pos, dep_plus_2.pos, dep_plus_1.pos),
        ]
    }
    return head_features, dep_features


def shape(word):
    """Inspired by spaCy's `token.shape` attribute."""
    punct = (',', '.', ';', ':', '?', '!')
    special = ('-', '/', '@', '#', '$', '%', '&')
    shape = ''
    for char in word:
        if char.isupper():
            shape += 'X'
        elif char.islower():
            shape += 'x'
        elif char.isdigit():
            shape += 'd'
        elif char in punct or char in special:
            shape += char
        else:
            shape += 'c'
    return shape


def get_features(head, dep, line, complex=False):
    def get_left(line, id):
        if id > -1:
            token = line[id]
        else:
            token = START_TOKEN
        return token

    def get_right(line, id):
        try:
            token = line[id]
        except IndexError:
            token = END_TOKEN
        return token

    dep_min_2 = get_left(line, dep.id - 2)
    dep_min_1 = get_left(line, dep.id - 1)
    head_min_2 = get_left(line, head.id - 2)
    head_min_1 = get_left(line, head.id - 1)

    dep_plus_1 = get_right(line, dep.id + 1)
    dep_plus_2 = get_right(line, dep.id + 2)
    head_plus_1 = get_right(line, head.id + 1)
    head_plus_2 = get_right(line, head.id + 2)
    if complex:
        head_features, dep_features = feature_dicts(
            dep,
            head,
            dep_min_2,
            dep_min_1,
            head_min_2,
            head_min_1,
            dep_plus_1,
            dep_plus_2,
            head_plus_1,
            head_plus_2
        )
        # Make feature conjunctions.
        self_features = [f'{h} + {d}' for h, d in zip(head_features['self'], dep_features['self'])]
        left_features = [f'{h} + {d}' for h, d in zip(head_features['left'], dep_features['left'])]
        right_features = [f'{h} + {d}' for h, d in zip(head_features['right'], dep_features['right'])]
        full_features = [f'{h} + {d}' for h, d in zip(head_features['full'], dep_features['full'])]
        features = tuple(self_features + left_features + right_features + full_features)
    else:
        # A super simple head-dep feature-set without any context.
        features = (
            'bias',
            'head dep word={} {}'.format(head.form, dep.form),
            'head dep first={} {}'.format(head.form[0], dep.form[0]),
            'head dep final={} {}'.format(head.form[-1], dep.form[-1]),
            'head dep suffix={} {}'.format(head.form[-3:], dep.form[-3:]),
            'head dep pref={} {}'.format(head.form[:3], dep.form[:3]),
            'head dep shape={} {}'.format(shape(head.form), shape(dep.form)),
            'head dep pos={} {}'.format(head.pos, dep.pos),
            'head dep id={} {}'.format(head.id, dep.id),
            'distance={}'.format(dep.id - head.id),
            )
    return features
