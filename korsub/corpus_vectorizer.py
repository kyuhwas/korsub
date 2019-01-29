from collections import defaultdict

def scan_subwords(lr_format_sents, min_count=10):
    """
    Arguments
    ---------
    lr_format_sent : list of list of (L,R) tuples

        For example,

            for sent in lr_format_sents:
                print(sent)

            $ [('실내', ''), ('장식용', ''), ('직물', ''), ('디자이너', '로'), ('나섰', '다')]
            $ [('컬렉션', '이라는'), ('이름', '으로'), ('전시회', '를'), ('열', '었다')]

    min_count : int
        Minumum occurrence of subword

    Returns
    -------
    idx_to_l : list of str
        idx to subword (str)
    l_to_idx : dict
        subword (str) to idx
    lsubs : dict
        subword counter. for example, lsub['이번'] = 3
    idx_to_r : list of str
        idx to subword (str)
    r_to_idx : dict
        subword (str) to idx
    rsubs : dict
        subword counter. for example, lsub['이번'] = 3

    Usage
    -----
    idx_to_l, l_to_idx, lsubs, idx_to_r, r_to_idx, rsubs = scan_subwords(corpus)
    """

    lsubs = defaultdict(int)
    rsubs = defaultdict(int)
    for lrs in lr_format_sents:
        for l, r in lrs:
            lsubs[l] += 1
            if r:
                rsubs[r] += 1
    lsubs = {sub:c for sub, c in lsubs.items() if c >= min_count}
    rsubs = {sub:c for sub, c in rsubs.items() if c >= min_count}

    idx_to_l = [l for l in sorted(lsubs, key=lambda x:-lsubs[x])]
    idx_to_r = [r for r in sorted(rsubs, key=lambda x:-rsubs[x])]
    l_to_idx = {l:idx for idx, l in enumerate(idx_to_l)}
    r_to_idx = {r:idx for idx, r in enumerate(idx_to_r)}

    return idx_to_l, l_to_idx, lsubs, idx_to_r, r_to_idx, rsubs

def lr_sents_to_features(lrs, lsubs, rsubs, check=False):
    """
    Arguments
    ---------
    lrs : list of tuple
        For example,

            lrs = [
                ('컬렉션', '이라는'),
                ('이름', '으로'),
                ('전시회', '를'),
                ('열', '었다')
            ]

    lsubs : set or dict of str
        Dictionary of L subwords
    rsubs : set or dict of str
        Dictionary of R subwords
    check : Boolean
        If True, use only known (included in lsubs or rsubs) subwords

    Returns
    -------
    list of tuple, (word, features)

    Usage
    -----

        word_and_features = lr_sents_to_features(
            lrs[-4:], lsubs=None, rsubs=None, check=False)

        for word, features in word_and_features:
            print('word = {}, features = {}'.format(word, features))

        $ word = ('컬렉션', 'L'), features = {('이라는이름으로', 1), ('이라는이름', 1), ('이라는', 1)}
          word = ('이라는', 'R'), features = {('이름으로', 1), ('이름', 1), ('컬렉션', -1)}
          word = ('이름', 'L'), features = {('으로전시회를', 1), ('으로전시회', 1), ('으로', 1), ('이라는', -1), ('컬렉션이라는', -1)}
          word = ('으로', 'R'), features = {('컬렉션이라는이름', -1), ('이름', -1), ('이라는이름', -1), ('전시회를', 1), ('전시회', 1)}
          word = ('전시회', 'L'), features = {('으로', -1), ('이름으로', -1), ('를열었다', 1), ('를', 1), ('를열', 1)}
          word = ('를', 'R'), features = {('으로전시회', -1), ('이름으로전시회', -1), ('열었다', 1), ('전시회', -1), ('열', 1)}
          word = ('열', 'L'), features = {('었다', 1), ('전시회를', -1), ('를', -1)}
          word = ('었다', 'R'), features = {('전시회를열', -1), ('를열', -1), ('열', -1)}
    """
    word_and_features = []

    max_i = len(lrs) - 1
    for i, (l, r) in enumerate(lrs):
        if i == 0:
            ll, lr = '', ''
        else:
            ll, lr = lrs[i-1][0], lrs[i-1][1]
        if i == max_i:
            rl, rr = '', ''
        else:
            rl, rr = lrs[i+1][0], lrs[i+1][1]

        if check:
            if not (ll in lsubs):
                ll = ''
            if not (lr in rsubs):
                lr = ''
            if not (rl in lsubs):
                rl = ''
            if not (rr in rsubs):
                rr = ''

        # features of L
        l_features = set()
        if i > 0:
            l_features.update({(sub, -1) for sub in [lr, ll+lr] if sub})
        if i == max_i and r:
            l_features.add((r, 1))
        else:
            l_features.update({(sub, 1) for sub in [r, r+rl, r+rl+rr] if sub})

        if l_features:
            word_and_features.append(((l, 'L'), l_features))

        if not r:
            continue

        # features of R
        r_features = set()
        if i == 0:
            r_features.add((l, -1))
        else:
            r_features.update({(sub, -1) for sub in [l, lr+l, ll+lr+l] if sub})
        if i < max_i:
            r_features.update({(sub, 1) for sub in [rl, rl+rr] if sub})

        if r_features:
            word_and_features.append(((r, 'R'), r_features))

    return word_and_features