import os
import re
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

class TrainCorpus:
    """
    Arguments
    ---------
    path : str
        Corpus path
    xsv_as_adj : Boolean
        If True, use LR 0 else LR 1
    num_sent : int
        Maximum number of sent to be yield
        Default is -1 (use all)

    Usage
    -----
        train_corpus = TrainCorpus(path)
        for sent in train_corpus:
            # do something

    Description
    -----------
    Tap separated four columns. <어절, 세종 말뭉치의 형태소 품사, LR 0, LR 1>

    LR 0 은 NNG + VCP 를 형용사로 봅니다.

        eg) 컬렉션/NNG + 이/VCP + 라는/ETM --> 컬렉션이/Adjective + 라는/Eomi

    LR 1 은 NNG + VCP + ETM 을 명사 + 조사로 봅니다.

        eg) 컬렉션/NNG + 이/VCP + 라는/ETM --> 컬렉션/Noun + 이라는/Josa

    세종 말뭉치 예시

        컬렉션이라는	컬렉션/NNG '/SS 이/VCP 라는/ETM	컬렉션이/Adjective 라는/Eomi	컬렉션/Noun 이라는/Josa
        이름으로	이름/NNG 으로/JKB	이름/Noun 으로/Josa	이름/Noun 으로/Josa
        전시회를	전시회/NNG 를/JKO	전시회/Noun 를/Josa	전시회/Noun 를/Josa
        열었다.	열/VV 었/EP 다/EF ./SF	열/Verb 었다/Eomi	열/Verb 었다/Eomi

        목욕가운부터	목욕/NNG 가운/NNG 부터/JX	목욕가운/Noun 부터/Josa	목욕가운/Noun 부터/Josa
        탁자보,	탁자보/NNG ,/SP	탁자보/Noun	탁자보/Noun
        냅킨,	냅킨/NNG ,/SP	냅킨/Noun	냅킨/Noun

    """

    def __init__(self, paths, xsv_as_adj=False, num_sent=-1):
        if isinstance(paths, str):
            paths = [paths]
        self.paths = paths
        self.xsv_as_adj = xsv_as_adj
        self._col = 2 if xsv_as_adj else 3
        self.num_sent = num_sent

    def __iter__(self):
        def parse(morphtags):
            lr = morphtags.split()
            morph0, tag0 = lr[0].rsplit('/', 1)
            if len(lr) == 2:
                morph1, tag1 = lr[1].rsplit('/', 1)
            else:
                morph1, tag1 = '', ''
            return ((morph0, tag0), (morph1, tag1))

        def normalize(s):
            pattern = re.compile('[^가-힣]')
            return pattern.sub('', s)

        sent = []
        n_sents = 0
        for path in self.paths:
            with open(path, encoding='utf-8') as f:
                for doc in f:
                    if self.num_sent > 0 and n_sents >= self.num_sent:
                        break
                    doc = doc.strip()

                    if not doc:
                        yield sent
                        n_sents += 1
                        sent = []
                        continue

                    cols = doc.split('\t')
                    eojeol = cols[0]
                    morphtags = cols[self._col]
                    sent.append((normalize(eojeol), parse(morphtags)))

                if sent:
                    yield sent