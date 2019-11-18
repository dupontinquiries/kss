class k_chunk:

    DEFAULT_FLOOR = -500
    v = 0
    sv = 0
    data = []
    timestamp = None

    def __init__(self, i=0, l=[], sl=1, sr=1, t_s=0, t_f=1, source=None, dud=False):
        self.data = list()
        self.i = i
        if dud:
            self.v = 0
            self.sv = 0
        else:
            self.v = self.floor_out(l[i].dBFS, self.DEFAULT_FLOOR)
            self.sv = self.gen_sv(sl, sr, l, i)
        self.t_s = t_s
        self.t_f = t_f
        self.d = t_f - t_s

        self.data.append(self.v)
        self.data.append(self.sv)

        self.timestamp = (self.t_s, self.t_f)

        self.source = source

    def gen_sv(self, sl, sr, l, i):
        t = 0
        n = 0
        for o in range(max(0, i - sl), min(len(l) - 1, i + sr)):
            add = self.floor_out(l[o].dBFS, self.DEFAULT_FLOOR)
            t += add
            n += 1
        avg = t / n
        return avg


    def floor_out(self, a, bottom):
        if a < bottom:
            return bottom
        else:
            return a


    def __repr__(self):
        return repr('[CHUNK] @ {0}, v = {1}, sv = {2}'.format(self.timestamp, self.v, self.sv))


    def __setitem__(self, n, data):
          self.data[n] = data


    def __getitem__(self, n):
          return self.data[n]


    def __eq__(self, b):
        return (self.source == b.source \
            and self.t_s == b.t_s \
            and self.t_f == self.t_f \
            and self.v == b.v \
            and self.sv == b.sv)


    def v():
        return self.v


    def sv():
        return self.sv


    def t():
        return (self.t_s, self.t_f)
