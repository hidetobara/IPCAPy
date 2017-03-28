import numpy
import IncrementalPCA


class IncrementalPCAConvolution:
    Size = []
    _ipca = None

    def __init__(self, axis, size):
        self.Size = size
        length = 1
        for v in size: length *= v
        self._ipca = IncrementalPCA.IncrementalPCA(axis, length)

    def load(self, path):
        self._ipca.load(path)

    def save(self, path):
        self._ipca.save(path, self.Size)

    def fit(self, mat):
        mheight = mat.shape[0]
        mwidth = mat.shape[1]
        bheight = self.Size[0]
        bwidth = self.Size[1]
        for h in range(mheight - bheight):
            for w in range(mwidth - bwidth):
                trimmed = mat[h:h+bheight, w:w+bwidth, ]
                self._ipca.fit(trimmed)

    def transform(self, mat):
        mheight = mat.shape[0]
        mwidth = mat.shape[1]
        bheight = self.Size[0]
        bwidth = self.Size[1]
        array = []
        for h in range(mheight - bheight):
            for w in range(mwidth - bwidth):
                transformed = self._ipca.transform(mat[h:h+bheight, w:w+bwidth, ])
                array.append(transformed)
        return numpy.asarray(array).reshape((mheight-bheight,mwidth-bwidth,self._ipca.Axis))
