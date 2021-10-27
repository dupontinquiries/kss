import os
import shutil

import subprocess

class kPath:
    def __init__(self, p):
        if isinstance(p, kPath):
            p = p.aPath()
        self.p = os.path.abspath(p)
        if not os.path.exists(self.p) and self.p[-4] != '.':
            os.mkdir(self.p)


    def __eq__(self, b):
        return self.p == b.p


    def delete(self):
        if self.p[-4] != '.':
            import shutil
            shutil.rmtree(self.p)
        elif os.path.exists(self.p): # and self.p[-4] == '.':
            os.remove(self.p)


    def chop(self):
        v = kPath('\\'.join(self.p.split('\\')[:-1]))
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
        return kPath(v)


    def path(self):
        return self.p.split('\\')[-1]


    def aPath(self):
        return self.p


    def isFile(self):
        return os.path.isfile(self.p)


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


    def getDuration(self):
        if self.p[-4:] in extList or self.p[-5:] in extList:
            result = subprocess.Popen(["ffprobe", self.p], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return [x for x in result.stdout.readlines() if "Duration" in x]
        else:
            return -1


def kIn(a):
    b = input(a)
    print(b)
    return b

#error in dir M:\2019\Recordings 2019\GoPro\footage\2019-07-10\HERO5 Black 1 because of second layer of photos

def kFileCharacteristic(filename, ret='all'):
    cmd = 'ffprobe "{0}" -show_format ' \
        .format(filename)
    result = subprocess \
        .Popen(cmd, \
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    results = result.communicate()
    #print(results)
    w = None
    if ret == 'all':
        w = results
    else:
        try:
            w = float(str(results).split(ret)[1].split('\\')[0].replace('=', ''))
        except:
            return False
    result.kill()
    return w


def compressDir(cmd, fTypes, root, inD, outD, completedConversions, failedConversions, subF):
    """
    root = hard path of parent folder
    inD  = directory of videos to compress; changes with recursion
    outD = output folder to deliver to; constant with recursion
    fs   = file size limit to deliver to ffmpeg processing
    """
    for name in os.listdir(inD.aPath()):
        fPath = kPath(inD.append(name))
        if fPath.isFile():
            nameRoot = name[:-4]
            nameExt = name[-4:]
            if nameExt.lower() not in fTypes:
                continue
            if fPath.aPath() in completedConversions:
                continue
            out = outD
            if subF != '':
                out = out.append(subF)
            if not out.exists():
                os.mkdir(out.aPath())
            cmd_ = cmd.replace('(f)', fPath.aPath()).replace('(fr)', nameRoot).replace('(fe)', nameExt)
            print(nameRoot)
            print(cmd_)
            os.system('{0}'.format(cmd_))
            completedConversions.add(fPath.aPath())
        else:
            subFB = subF
            if subF != '':
                subF = subF + '\\'
            subF = subF + name
            compressDir(cmd, root, inD.append(name), outD, completedConversions, failedConversions, subF)
            subF = ''


root = kPath(kIn('Path to the workspace => '))

cmd = kIn('cmd to run with (f) for filename and (fr), (fe) for root & extension => ')
fTypes = (kIn('file types, separated by spaces, to select for => ')).strip().split(' ')
fTypes = list(filter(lambda x: len(x) > 0, fTypes))

outD = kPath(root.append('../program_results'))

if not outD.exists():
    outD.make()

os.chdir(root.aPath())

completedConversions = set()
failedConversions = set()

compressDir(cmd, fTypes, root, root, outD, set(), set(), '')
