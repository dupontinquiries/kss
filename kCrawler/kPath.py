import os

class kPath:
    def __init__(self, p):
        if isinstance(p, kPath):
            p = p.aPath()
        self.p = os.path.abspath(p)
        print(self.p)
        #if not os.path.exists(self.p) and self.p[-4] != '.':
            #os.mkdir(self.p)
            #print('had to make the directory "{0}"'.format(self.p))


    def __eq__(self, b):
        return self.p == b.p


    def chop(self):
        v = kPath('\\'.join(self.p.split('\\')[:-1]))
        #print('chopping from: "{0}"\n to: "{1}"'.format(self.p, v))
        return kPath(v)


    def cascadeCreate(self, p):
        pChunks = p.split('\\')
        s = pChunks[0]
        end = len(pChunks)
        for i in range(1, end):
            s += '\\' + pChunks[i]
            if s[-4] == '.' or 'mp4' in p:
                continue
            elif not os.path.exists(s):
                os.mkdir(s)


    def append(self, w):
        v = self.p + '\\' + w
        return kPath(v)


    def make(self):
        os.mkdir(self.p)


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


    def isFolder(self):
        return not os.path.isfile(self.p)


    def isDir(self):
        return os.path.isdir(self.p)


    def __repr__(self):
        return self.p


    def __str__(self):
        return self.p


    def exists(self):
        return os.path.exists(self.p)
