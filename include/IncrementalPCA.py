import numpy
import os
from PIL import Image


class IncrementalPCA:
    """
    Run Incremental PCA
    MERIT: It consumes very low memory, maybe Axis * Size * N.
    DEMRIT: It takes long time.
    """
    Length = None
    Axis = 32
    Amnesic = 0.1
    _main = None # (Axis,Size)
    _sub = None
    _inv_norm = None # (Size,1)
    _frame = 0

    def __init__(self, axis, length, amnesic=0.1):
        self.Length = length
        self.Axis = axis
        self.Amnesic = amnesic
        self._main = numpy.zeros((self.Axis, self.Length))
        self._sub = numpy.zeros((self.Axis, self.Length))

    def save(self, path, size=None):
        f = open(path + ".def", "w")
        f.write("axis=" + str(self.Axis) + "\n")
        f.write("size=" + str(self.Length) + "\n")
        f.write("frame=" + str(self._frame) + "\n")
        f.close()
        numpy.save(path + ".main.npy", self._main)
        numpy.save(path + ".sub.npy", self._sub)

        if isinstance(size, tuple) and len(size) == 3:
            for a in range(self.Axis):
                low = min(self._main[a])
                high = max(self._main[a])
                diff = high - low
                # print(low, high, limit, diff)
                main = numpy.uint8((self._main[a] - low) * 255.0 / diff)
                stride = size[1] * size[2]
                img = Image.new("RGB", size[0:2])
                for h in range(size[0]):
                    for w in range(size[1]):
                        index = stride * h + w * size[2]
                        img.putpixel((w,h), tuple(main[index:index+3])) # first 3 columns only
                name = "%02d" % a
                img.save(path + "." + name + ".png")

    def load(self, path):
        if os.path.exists(path + ".def") is False: return

        for line in open(path + ".def", "r"):
            try:
                cells = line.strip().split("=")
                if cells[0] == "axis": self.Axis = int(cells[1])
                if cells[0] == "size": self.Length = int(cells[1])
                if cells[0] == "frame": self._frame = int(cells[1])
            except Exception as ex:
                print(str(ex))
        self._main = numpy.load(path + ".main.npy")
        self._sub = numpy.load(path + ".sub.npy")
        if self._main.shape[0] != self.Axis or self._main.shape[1] != self.Length:
            raise Exception("Axis, Size id different.")

    def prepare_inv_norm(self):
        if self._frame < self.Axis: return
        invs = []
        for a in range(self.Axis):
            invs.append(1.0 / numpy.linalg.norm(self._main[a]))
        self._inv_norm = numpy.asarray(invs, dtype=numpy.float64).reshape((self.Axis, 1))

    def fit(self, row):
        row = row.reshape((self.Length))
        self._sub[0] = row

        iterate = self.Axis - 1
        if self.Axis > self._frame:
            iterate = self._frame

        for i in range(iterate + 1):
            if i == self._frame:
                self._main[i] = self._sub[i]
                continue

            # Vi(n) = [a= (n-1-l)/n * Vi(n-1)] + [b= (1+l)/n * Ui(n)T Vi(n-1)/|Vi(n-1)| * Ui(n) ]
            nrmV = numpy.linalg.norm(self._main[i])
            scalerA = (self._frame - 1.0 - self.Amnesic) / self._frame
            dotUV = numpy.dot(self._sub[i], self._main[i])
            scalerB = (1.0 + self.Amnesic) * dotUV / (self._frame * nrmV)

            self._main[i] = self._main[i] * scalerA + self._sub[i] * scalerB
            if i == iterate: continue

            # ///// Ui+1(n) = Ui(n) - [c= Ui(n)T Vi(n)/|Vi(n)| * Vi(n)/|Vi(n)| ]
            nrmV = numpy.linalg.norm(self._main[i])
            dotUV = numpy.dot(self._sub[i], self._main[i])
            scalerC = dotUV / (nrmV * nrmV)

            self._sub[i + 1] = self._sub[i] - self._main[i] * scalerC

        if self._frame > 0 and self._frame % 1000 == 0:
            print("\tframe=" + str(self._frame))
        self._frame += 1
        self._inv_norm = None

    def transform(self, row):
        """
        射影します
        :param row: 大きさは(1,Size)
        :return: 大きさは(Axis,1)
        """
        row = row.reshape((self.Length, 1))
        if self._inv_norm is None:
            self.prepare_inv_norm()
        return self._main.dot(row) * self._inv_norm * self._inv_norm

    def inv_transform(self, res):
        """
        逆射影します
        :param res: 大きさは(Axis,1)
        :return: 大きさは(1,Size)
        """
        if self._inv_norm is None:
            self.prepare_inv_norm()
        #res = res * self._inv_norm
        return res.transpose().dot(self._main)

    def evaluate_outlier(self, row, is_refreshing=False):
        """
        与えられた要素がどれだけ、基底から外れているかを計算
        """
        if self._frame < self.Axis: return None
        if is_refreshing: self._inv_norm = None

        row = row.reshape((1, self.Length))
        compress = self.transform(row)
        restored = self.inv_transform(compress)
        return numpy.linalg.norm(row - restored) / self.Length

    def evaluate_basis(self):
        """
        基底がどれだけ独立かを計算
        """
        self.prepare_inv_norm()
        values = []
        for i in range(self.Axis):
            for j in range(self.Axis):
                values.append(self._main[i].dot(self._main[j]) * self._inv_norm[i] * self._inv_norm[j])
        return numpy.asarray(values, dtype=numpy.float64).reshape((self.Axis,self.Axis))
