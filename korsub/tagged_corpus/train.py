from .vectorizer import scan_subwords
from .vectorizer import scan_features
from .vectorizer import count_word_features
from ..math import train_pmi
from ..math import train_svd
from ..utils import c_to_x

def train_lr2vec(lr_corpus, vocab_min_count=10, feature_min_count=5,
    min_cooccurrence=2, prune_per_sent=100000, min_pmi=0,
    beta=0.75, n_components=300, verbose=True):

    idx_to_l, l_to_idx, _, idx_to_r, r_to_idx, _ = scan_subwords(
        lr_corpus, vocab_min_count)

    if verbose:
        print('num of L = {}, R = {} with min count = {}'.format(
            len(idx_to_l), len(idx_to_r), vocab_min_count))

    idx_to_feature, feature_to_idx = scan_features(
        lr_corpus, l_to_idx, r_to_idx, feature_min_count)

    sub_dic = {sub for sub in idx_to_l}
    sub_dic.update(idx_to_r)

    C = count_word_features(lr_corpus, sub_dic,
        feature_to_idx, min_cooccurrence, prune_per_sent)

    X, idx_to_row, idx_to_col = c_to_x(C)

    pmi, px, py = train_pmi(X, min_pmi = min_pmi, beta = beta)
    U, Sigma, VT = train_svd(pmi, n_components)
    wv = U * (Sigma ** (0.5))
    mapper = VT.T * (Sigma ** (-0.5))

    row_to_idx = {row:idx for idx, row in enumerate(idx_to_row)}

    return X, idx_to_row, idx_to_col, pmi, py, wv, mapper
