import numpy

class IncrementalPCA:
    Size = None
    Axis = 32
    Amnesic = 0.2
    _Main = None
    _Sub = None
    _Frame = 0

    def __init__(self, size, axis, amnesic=0.2):
        self.Size = size
        self.Axis = axis
        self.Amnesic = amnesic
        _Main = numpy.zeros((self.Axis, self.Size))
        _Sub = numpy.zeros((self.Axis, self.Size))

    def save(self, path):
        f = open(path + ".def", "w")
        f.write("axis=" + self.Axis + "\n")
        f.write("size=" + self.Size + "\n")
        f.write("frame=" + self._Frame + "\n")
        f.close()
        self._Main.save(path + ".main.npy")
        self._Sub.save(path + ".sub.npy")

    def load(self, path):
        for line in open(path + ".def", "r"):
            try:
                cells = line.strip().split("=")
                if cells[0] == "axis": self.Axis = int(cells[1])
                if cells[1] == "size": self.Size = int(cells[1])
                if cells[2] == "frame": self._Frame = int(cells[1])
            except Exception as ex:
                print(str(ex))
        self._Main = numpy.load(path + ".main.npy")
        self._Sub = numpy.load(path + ".sub.npy")

    def fit(self, row):
        self._Sub[0] = row.copy()

        iterate = self.Axis - 1
        if self.Axis > self._Frame:
            iterate = self._Frame

        for i in range(iterate + 1):
            if i == self._Frame:
                self._Main[i] = self._Sub[i]
                continue

            # Vi(n) = [a= (n-1-l)/n * Vi(n-1)] + [b= (1+l)/n * Ui(n)T Vi(n-1)/|Vi(n-1)| * Ui(n) ]
            nrmV = numpy.linalg.norm(self._Main[i])
            scalerA = (self._Frame - 1.0 - self.Amnesic) / self._Frame
            dotUV = numpy.linalg.dot(self._Sub[i], self._Main[i])
            scalerB = (1.0 + self.Amnesic) * dotUV / (self._Frame * nrmV)

            self._Main[i] = self._Main[i] * scalerA + self._Sub[i] * scalerB
            if i == iterate: continue

            # ///// Ui+1(n) = Ui(n) - [c= Ui(n)T Vi(n)/|Vi(n)| * Vi(n)/|Vi(n)| ]
            nrmV = numpy.linalg.norm(self._Main[i])
            dotUV = numpy.linalg.dot(self._Sub[i], self._Main[i])
            scalerC = dotUV / (nrmV * nrmV)

            self._Sub[i+1] = self._Sub[i] - self._Main[i] * scalerC

            if self._Frame % 1000 == 0: print("\tframe=" + str(self._Frame))
            self._Frame += 1

    def transform(self, row):
        return self._Main.dot(numpy.linalg.transpose(row))
