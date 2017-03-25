import IncrementalPCA


class IncrementalPCAConvolution:
    Size = []
    _ipca = None

    def __init__(self, axis, size):
        self.Size = size
        length = 1
        for v in size: length *= v
        self._ipca = IncremtalPCA.IncremtalPCA(axis, length)

    def fit(self, mat):
        pass

    def transform(self, row):
        pass
