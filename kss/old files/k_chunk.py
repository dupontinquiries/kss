class k_chunk:

    DEFAULT_FLOOR = -500
    v = 0
    vs = 0

    def __init__(self, i=0, l=[], sl=1, sr=1, t_s=0, t_f=1, dud=False):
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

    def gen_sv(self, sl, sr, l, i):
        t = 0
        n = 0
        for o in range(max(0, i - sl), min(len(l) - 1, i + sr)):
            add = self.floor_out(l[o].dBFS, self.DEFAULT_FLOOR)
            t += add
            n += 1
        avg = t // n
        return avg

    def floor_out(self, a, bottom):
        if a < bottom:
            return bottom
        else:
            return a

    def __repr__(self):
        return repr(('[CHUNK]' + 'v: ' + str(self.v) + ' sv: ' + str(self.sv)))

    def __setitem__(self, key, val):
        self.__setitem__(self, key, val)

    def __getitem__(self, key):
        return self.__getitem__(self, key)

    def __getattr__(self, attr):
        return attr

    def v(self):
        return self.v

    def sv(self):
        return self.sv
