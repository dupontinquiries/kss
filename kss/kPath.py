import os

class kPath:
    def __init__(self, p):
        if isinstance(p, kPath):
            p = p.aPath()
        self.p = os.path.abspath(p)
        if not os.path.exists(self.p) and os.path.isdir(self.p):
            os.mkdir(self.p)
            print('had to make the directory "{0}"'.format(self.p))


    def __eq__(self, b):
        return self.p == b.p


    def chop(self):
        v = kPath('\\'.join(self.p.split('\\')[:-1]))
        #print('chopping from: "{0}"\n to: "{1}"'.format(self.p, v))
        return kPath(v)


    def append(self, w):
        v = self.p + '\\' + w
        #print('appending "{0}" \nto get this: "{1}"'.format(w, v))
        return kPath(v)


    def hitch(self, w):
        v = self.p + w
        #print('hitching "{0}" \nto get this: "{1}"'.format(w, v))
        return kPath(v)


    def path(self):
        return self.p.split('\\')[-1]


    def aPath(self):
        return self.p


    def isFile(self):
        return os.path.isfile(self.p)
        #self.p[-4] == '.' and


    def isDir(self):
        return os.path.isdir(self.p)


    def __repr__(self):
        return self.p


    def __str__(self):
        return self.p


    def exists(self):
        return os.path.exists(self.p)
