import os
import psutil
from sklearn.metrics import pairwise_distances


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

class Sentences:
    def __init__(self, path, num_sent=-1, lowercase=True, verbose_point=-1):
        self.path = path
        self.num_sent = num_sent
        self.lowercase = lowercase
        self.verbose_point = verbose_point
        self._len = 0
        self._num_iter = 0

    def __iter__(self):
        vp = self.verbose_point
        with open(self.path, encoding='utf-8') as f:
            for i, sent in enumerate(f):
                if self.num_sent > 0 and i >= self.num_sent:
                    break
                if vp > 0 and i % vp == 0:
                    print('\riter = %d, num sents = %d%s' % (self._num_iter, i, ' '*15), end='')
                sent = sent.strip()
                if not sent:
                    continue
                yield sent.split()
            self._len = i + 1
        if vp > 0:
            print('\r%d th iterating was done. num sents = %d%s' % (self._num_iter, i+1, ' '*15))
        self._num_iter += 1

    def __len__(self):
        if self._len == 0:
            with open(self.path, encoding='utf-8') as f:
                for i, _ in enumerate(f):
                    continue
                self._len = (i+1)
        return self._len

    def reset_num_iter(self):
        self._num_iter = 0