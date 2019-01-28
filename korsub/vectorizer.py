from collections import defaultdict
from scipy.sparse import csr_matrix
from .utils import get_process_memory

def scan_subwords(sentences, submax=5, min_count=10,
    prune_per_sent=2000000, prune_min_count=2, verbose=True):

    subwords = {}
    features = {}

    prune = lambda d, m:{k:v for k,v in d.items() if v >= m}
    num_subwords = lambda: len(subwords)
    num_features = lambda: len(features)

    def status(i_sent, newline=False):
        form = '\rscan {} subwords, {} features, from {} sents, mem = {:.3} GB{}'
        message = form.format(
            num_subwords(), num_features(), i_sent, get_process_memory(), ' '*20)
        print(message, end='\n' if newline else '')

    for i_sent, words in enumerate(sentences):
        if prune_per_sent > 0 and i_sent % prune_per_sent == 0:
            features = prune(features, prune_min_count)
            subwords = prune(subwords, prune_min_count)

        for word in words:
            n = len(word)
            if n <= 1:
                continue
            for i in range(2, n + 1):
                l = word[:i] if i <= submax else None
                r = word[i:] if (n - i) < submax else None
                if l is not None:
                    features[l] = features.get(l, 0) + 1
                if r is not None:
                    features[r] = features.get(r, 0) + 1
            for i in range(2, n + 1):
                sub = word[:i]
                subwords[sub] = subwords.get(sub, 0) + 1

        if verbose and i_sent % 10000 == 0:
            status(i_sent)

    subwords = prune(subwords, min_count)
    if verbose:
        status(i_sent+1, newline=True)

    return subwords, features

def enumerate_r_parts(word, submax, dic):
    for i in range(1, min(submax, len(word)) + 1):
        sub = word[-i:]
        if sub in dic:
            yield sub

def enumerate_l_parts(word, submax, dic):
    for i in range(2, min(submax, len(word)) + 1):
        sub = word[:i]
        if sub in dic:
            yield sub

def prune(C, min_count):
    C = {k1:{k2:v for k2, v in d.items() if v >= min_count} for k1, d in C.items()}
    C = {k1:defaultdict(int, d) for k1, d in C.items() if d}
    C = defaultdict(lambda: defaultdict(int), C)
    return C

def subword_features(sentences, subwords, subfeatures, min_count=2,
    prune_per_sent=1000000, prune_min_count=2, verbose=True):

    C = defaultdict(lambda: defaultdict(int))

    for i_sent, words in enumerate(sentences):
        if i_sent % prune_per_sent == 0:
            C = prune(C, prune_min_count)

        if verbose and i_sent % 10000 == 0:
            print('\rcounting co-occurrence from {} sents, mem={:.3} Gb{}'.format(
                i_sent, get_process_memory(), ' '*5), end='')

        n_words = len(words)

        for i_word, word in enumerate(words):

            n = len(word)
            if n == 1:
                continue

            for e_sub in range(2, n + 1):
                subword = word[:e_sub]
                if not (subword in subwords):
                    continue

                # leftside features
                if i_word > 0:
                    for left in enumerate_r_parts(words[i_word-1], 5, subfeatures):
                        C[subword][(0, left)] += 1

                r = word[e_sub:]
                if r and not (r in subfeatures):
                    continue

                # r features
                if r:
                    C[subword][(1, r)] += 1

                # rightside features
                if i_word < n_words - 1:
                    for right in enumerate_l_parts(words[i_word+1], 5, subfeatures):
                        C[subword][(1, r + right)] += 1

    C = {k1:{k2:v for k2, v in d.items() if v >= min_count} for k1, d in C.items()}
    C = {k1:d for k1, d in C.items() if d}

    if verbose:
        print('\rcounting co-occurrence from {} sents was done. mem={:.3} Gb{}'.format(
            i_sent, get_process_memory(), ' '*5))

    return C

def c_to_x(C):
    row_counter = {}
    col_counter = defaultdict(int)
    for r, d in C.items():
        row_counter[r] = sum(d.values())
        for c, v in d.items():
            col_counter[c] += v
    idx_to_row = [r for r in sorted(row_counter, key=lambda x:-row_counter[x])]
    idx_to_col = [c for c in sorted(col_counter, key=lambda x:-col_counter[x])]
    row_to_idx = {r:idx for idx, r in enumerate(idx_to_row)}
    col_to_idx = {c:idx for idx, c in enumerate(idx_to_col)}

    rows, cols, data = [], [], []
    for r, d in C.items():
        i = row_to_idx[r]
        for c, v in d.items():
            j = col_to_idx[c]
            rows.append(i)
            cols.append(j)
            data.append(v)

    X = csr_matrix((data, (rows, cols)))
    return X, idx_to_row, idx_to_col