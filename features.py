from utils import XToken

START = '<sos>'
END = '<eos>'

START_TOKEN = XToken(
    -1, START, START, START, START, START, -1, START, START, START)

END_TOKEN = XToken(
    -1, END, END, END, END, END, -1, END, END, END)


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


def get_features(head, dep, line, add_surrounding=True, add_distance=True, add_inbetween=True):
    """Feature-set loosely following McDonald et al. 2006."""
    def get_token(line, id):
        if id in range(len(line)):
            token = line[id]
        elif id < 0:
            token = START_TOKEN
        else:
            token = END_TOKEN
        return token

    dep_min_1 = get_token(line, dep.id - 1)
    head_min_1 = get_token(line, head.id - 1)
    dep_min_2 = get_token(line, dep.id - 2)
    head_min_2 = get_token(line, head.id - 2)

    dep_plus_1 = get_token(line, dep.id + 1)
    head_plus_1 = get_token(line, head.id + 1)
    dep_plus_2 = get_token(line, dep.id + 2)
    head_plus_2 = get_token(line, head.id + 2)

    # Basic arc features
    features = (
        'distance=%d' % (dep.id - head.id),  # a general distance bias
        'head dep word word=%s %s' % (head.form, dep.form),
        'head dep pos pos=%s %s' % (head.pos, dep.pos),

        'head dep word pos=%s %s' % (head.form, dep.pos),
        'head dep pos word=%s %s' % (head.pos, dep.form),

        'head dep first first=%s %s' % (head.form[:2], dep.form[:2]),
        'head dep final final=%s %s' % (head.form[:-2], dep.form[:-2]),

        'head dep suffix suffix=%s %s' % (head.form[-3:], dep.form[-3:]),
        'head dep pref pref=%s %s' % (head.form[:3], dep.form[:3]),

        'head dep shape shape=%s %s' % (shape(head.form), shape(dep.form)),
        'head dep id id=%d %d' % (head.id, dep.id),
        )

    if add_distance:
        # Stamp each basic features with a distance.
        features = tuple(f + ' (%d)' % (dep.id - head.id) for f in features[1:])  # distance does not need distance stamp

    if add_surrounding:
        features += (
            'head dep  0 +1\-1  0=%s %s\%s %s' % (head.pos, head_plus_1.pos, dep_min_1.pos, dep.pos),
            'head dep -1  0\-1  0=%s %s\%s %s' % (head_min_1.pos, head.pos, dep_min_1.pos, dep.pos),
            'head dep  0 +1\ 0 +1=%s %s\%s %s' % (head.pos, head_plus_1.pos, dep.pos, dep_plus_1.pos),
            'head dep -1  0\ 0 +1=%s %s\%s %s' % (head_min_1.pos, head.pos, dep.pos, dep_plus_1.pos)
        )

    if add_inbetween:
        betweens = line[head.id+1:dep.id]
        if not betweens:
            betweens = line[dep.id+1:head.id]
        features += tuple(
            ('head between dep=%s %s %s (%d %d)' % (
                head.pos, between.pos, dep.pos, between.id-head.id, dep.id-head.id)
                for between in betweens))
    return features
