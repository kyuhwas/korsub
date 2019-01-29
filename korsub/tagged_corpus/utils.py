import re


def to_lrs(eojeols, morphtags):
    lrs = []
    for e, mt in zip(eojeols, morphtags):
        i = len(mt[0][0])
        l, r = e[:i], e[i:]
        lrs.append((l, r))
    return lrs

def decorate_four_column_to_lr_sent(corpus):
    for sent in corpus:
        if not sent:
            continue
        eojeols, morphtags = zip(*sent)
        lrs = to_lrs(eojeols, morphtags)
        yield lrs


class FourColumnCorpus:
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

                    try:
                        cols = doc.split('\t')
                        eojeol = cols[0]
                        morphtags = cols[self._col]
                        sent.append((normalize(eojeol), parse(morphtags)))
                    except:
                        continue

                if sent:
                    yield sent