#  TODO:  add folder selection and auto folder creation
#      -  function that passes chunks to main class
#         make function that normalizes audio signal (use tensorflow tensors and raw audio data)
#         make function that con piece together audio and related video data


#imports

from __future__ import unicode_literals

#av

#video

import ffmpeg
from moviepy.video.io.ffmpeg_tools import *
import moviepy
import moviepy.editor as mpye

#audio

import cv2
from pydub import AudioSegment
from pydub.utils import make_chunks

#math

import statistics
import random

#readability

from termcolor import colored

#fs

import os
from os.path import dirname, abspath
import sys
import shutil
import subprocess

#my stuff

#from k_chunk import k_chunk
#from kPath import kPath

# functions

FFMPEG_BIN = 'ffmpeg'
dir = ''
dir = dirname(abspath(__file__)) + "\\footage"
chunks = []
gen = []

# def val presets
# car voice: 0.96, 500, 1, 0.94, 2560, 1440, 1
presetDefaults = \
{ \
'vodCast':
    [1.43, 390, 3, .99, 1280, 720, 10 * 60, 'voice'],
'vodCastPro':
    [1.35, 350, 4, .9, 1920, 1080, 10 * 60, 'voice'],
}

sp = presetDefaults['vodCast']
# default values
# processing
DEFAULT_THRESHOLD = sp[0]
DEFAULT_PERIOD = sp[1]
DEFAULT_REACH_ITER = sp[2]
DEFAULT_REACH_THRESH = sp[3] * DEFAULT_THRESHOLD
DEFAULT_WIDTH = sp[4]  # 2560
DEFAULT_HEIGHT = sp[5]  # 1440
DEFAULT_MAX_CHUNK_SIZE = sp[6] #1.2, 3.2, 10.2
DEFAULT_TREATMENT = sp[7]
#list(['voice', 'music'])[0]
verbose = False
cleanup = True
dir = ''
print_color = 'cyan'
print_color_error = 'yellow'
extList = ['.mp4', '.mov']


# id / random string generator
def randomString(stringLength=10):
    string = 'abcdefghijklmnopqrstuvwxyz1234567890'
    """Generate a random string of fixed length """
    letters = string
    return ''.join(random.choice(letters) for i in range(stringLength))


class kPath:
    def __init__(self, p):
        if isinstance(p, kPath):
            p = p.aPath()
        self.p = os.path.abspath(p
        if not os.path.exists(self.p) and self.p[-4] != '.':
            os.mkdir(self.p)


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
        if self.path()[-4:] in extList:
            result = subprocess.Popen(["ffprobe", filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return [x for x in result.stdout.readlines() if "Duration" in x]
        else return -1


    def chunk(self):
        if self.path()[-4:] in extList:
            list = list()
            d = self.getDuration()
            if d < 0:
                return None
            n = self.getDuration() // DEFAULT_MAX_CHUNK_SIZE
            video = mpye.VideoFileClip()
            for i in range(0, n - 1):
                list.append(
                    video.subclip(
                        i * DEFAULT_MAX_CHUNK_SIZE,
                        (i + 1) * DEFAULT_MAX_CHUNK_SIZE
                    )
                )
            list.append(
                (i + 1) * DEFAULT_MAX_CHUNK_SIZE,
                video.duration
            )
        return list
        else:
            return None


class k_chunk:

    DEFAULT_FLOOR = -1000
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
        return repr('[CHUNK] @ {0}, v = {1:.2f}, sv = {2:.2f}'.format(self.timestamp, self.v, self.sv))


    def __setitem__(self, n, data):
          self.data[n] = data


    def __getitem__(self, n):
          return self.data[n]


    def __eq__(self, b):
        return self.t_s == b.t_s and self.t_f == b.t_f


    def getV():
        return self.v


    def getSV():
        return self.sv


    def getT():
        return (self.t_s, self.t_f)


class kss2:

    sessID = None
    inD = None
    workD = None
    outD = None

    def __init__(self, sessID, inD, workD, outD = ''):
        self.sessID = sessID
        self.inD = inD
        self.workD = workD
        if len(outD) > 0:
            self.outD = self.inD.append('output')
        else:
            self.outD = outD
        vidList = self.vidList(inD)
        pvL = list()
        for v in vidList:
            pv = self.getProcessedVideo(v)
            grab = None
            if pv.getLength() > DEFAULT_MAX_CHUNK_SIZE:
                grab = pv.chunk()
            else:
                grab = pv
            pvL.append(grab)



    def vidList(self, k):
        fSet = list(os.listdir(k.aPath()))
        kF = list()
        for o in fSet:
            kF.append(k.append(o))
        vids = \
        sorted(
            list(

            #filter(lambda v: v.isFile ? , fSet)
            filter(lambda v: v[-4:].lower() in extList, fSet)

            )
        )
        return vids


    def getProcessedVideo(self, k):
        clips = list()
        if k.getDuration() > DEFAULT_MAX_CHUNK_SIZE:
            clips = k.chunk(DEFAULT_MAX_CHUNK_SIZE)
        else:
            clips.append(k)
        if len(clips) is 1:
            return k
        elif len(clips) is 0:
            return None
        else return clips


    def defaultGet(element):
        return element


    def getSV(element):
        return element.getSV()


    def getV(element):
        return element.getV()


    def normalize(self, list, fn):
        values = list()
        for o in list:
            values.append(fn(o))
        norm = np.linalg.norm(list)
        if norm == 0:
            return list
        return list / norm


    def splVal(list, threshold, fn):
        print()


if __name__ == '__main__':
    sessID = randomString()
    workD = kPath(dirname(abspath(__file__)) + "\\footage")
    inD = workD.append('input')
    outD = ''
    kss2(sessID, inD, workD, outD)
