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