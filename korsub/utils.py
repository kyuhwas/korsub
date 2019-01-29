from collections import defaultdict
import os
import psutil
from sklearn.metrics import pairwise_distances
from scipy.sparse import csr_matrix


def most_similar(query, wv, vocab_to_idx, idx_to_vocab, topk=10):
    """
    :param query: str
        String type query word
    :param wv: numpy.ndarray or scipy.sparse.matrix
        Topical representation matrix
    :param vocab_to_idx: dict
        Mapper from str type query to int type index
    :param idx_to_vocab: list
        Mapper from int type index to str type words
    :param topk: int
        Maximum number of similar terms.
        If set top as negative value, it returns similarity with all words
    It returns
    ----------
    similars : list of tuple
        List contains tuples (word, cosine similarity)
        Its length is topk
    """

    q = vocab_to_idx.get(query, -1)
    if q == -1:
        return []
    qvec = wv[q].reshape(1,-1)
    dist = pairwise_distances(qvec, wv, metric='cosine')[0]
    sim_idxs = dist.argsort()
    if topk > 0:
        sim_idxs = sim_idxs[:topk+1]
    similars = [(idx_to_vocab[idx], 1 - dist[idx]) for idx in sim_idxs if idx != q]
    return similars

def get_process_memory():
    """It returns the memory usage of current process"""
    
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 ** 3)

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