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