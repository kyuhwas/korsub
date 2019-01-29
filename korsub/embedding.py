from .tagged_corpus.train import train_lr2vec

class LR2Vec:
    def __init__(self, lr_corpus=None, vocab_min_count=20, feature_min_count=5,
        min_cooccurrence=2, beta=0.75, min_pmi=0.0, dim=300,
        prune_per_sent=1000000, verbose=True):

        self._vocab_min_count = vocab_min_count
        self._feature_min_count = feature_min_count
        self._min_cooccurrence = min_cooccurrence
        self._beta = beta
        self._min_pmi = min_pmi
        self._dim = dim
        self._prune_per_sent = prune_per_sent
        self.verbose = verbose

        if lr_corpus is not None:
            self.train(lr_corpus, self._vocab_min_count,
                self._feature_min_count, self._min_cooccurrence,
                self._prune_per_sent, self._min_pmi, self._beta,
                self._dim, self.verbose)

    def train(self, lr_corpus, vocab_min_count=10, feature_min_count=5,
        min_cooccurrence=2, prune_per_sent=100000, min_pmi=0,
        beta=0.75, n_components=300, verbose=True):

        returns = train_lr2vec(lr_corpus)

        self.X = returns[0]
        self.idx_to_row = returns[1]
        self.idx_to_col = returns[2]
        self.pmi = returns[3]
        self.py = returns[4]
        self.wv = returns[5]
        self.mapper = returns[6]

        if self.verbose:
            print('Train was done.')
