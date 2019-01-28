import pickle
from .utils import installpath

def load_lr_surface_9tags():
    """
    Returns
    -------
    X : scipy.sparse.csr_matrix
        (L, R) count sparse matrix
    idx_to_l : list of tuple
        Mapper from index to row value
    idx_to_r : list of str
        Mapper from index to column value
    idx_to_ltag : list of str
        Mapper from index to pos tag of row value
    idx_to_lmorph : list of str
        Mapper from index to morpheme of row value
    """

    param_path = '{}/LR_surface_9tags/params.pkl'.format(installpath)
    with open(param_path, 'rb') as f:
        params = pickle.load(f)

    X = params['X']
    idx_to_l = params['idx_to_l']
    idx_to_r = params['idx_to_r']
    idx_to_ltag = params['idx_to_ltag']
    idx_to_lmorph = params['idx_to_lmorph']

    return X, idx_to_l, idx_to_r, idx_to_ltag, idx_to_lmorph

def load_lr_surface_noun():
    raise NotImplemented