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

import tensorflow as tf
import numpy as np

#my stuff

#from kChunk import kChunk
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
'normalizedDefault':
    [.4, 180, 12, .9, 1920, 1080, 10 * 60, 'voice']
}

sp = presetDefaults['normalizedDefault']
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
        self.p = os.path.abspath(p)
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
        if self.p[-4:] in extList:
            result = subprocess.Popen(["ffprobe", self.p], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return [x for x in result.stdout.readlines() if "Duration" in x]
        else:
            return -1


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


    def getProcessedVideo(self):
        clips = list()
        k = mpye.VideoFileClip(self.p)
        if k.duration > DEFAULT_MAX_CHUNK_SIZE:
            clips = k.chunk(DEFAULT_MAX_CHUNK_SIZE)
        else:
            clips.append(k)
        if len(clips) is 1:
            return k
        elif len(clips) is 0:
            return None
        else:
            return clips


class kChunk:

    v = 0
    data = []

    def __init__(self, content, ts, tf, volume, sourceName):
        self.content = content
        self.ts = ts
        self.tf = tf
        self.timestamp = (self.ts, self.tf)
        self.volume = volume
        self.sourceName = sourceName
        self.data += [self.content, self.ts, self.tf, self.volume, self.sourceName]

    def __repr__(self):
        return repr('[CHUNK] @ {0}, v = {1:.3f}'.format(self.timestamp, self.volume))


    def __eq__(self, b):
        return self.t_s == b.t_s and self.t_f == b.t_f and self.sourceName == b.sourceName


class kss2:

    sessID = None
    inD = None
    workD = None
    outD = None

    def __init__(self, sessID, inD, workD, outD):
        self.sessID = sessID
        self.inD = inD
        self.workD = workD
        vidList = self.vidList(inD)
        import cv2
        from pydub import AudioSegment
        from pydub.utils import make_chunks
        import math
        import sys
        spreadCalc = DEFAULT_REACH_ITER // 2
        chuLenMS = DEFAULT_PERIOD
        chuLenS = chuLenMS / 1000
        apList = []
        videoChunks = []
        self.x = 100
        self.progress_x = 0
        self.title = 'chunking'
        self.startProgress()
        length = len(vidList)
        for i in range(length - 1): #now make this invigorate a list of kChunks so that the program can sticth video and audio in the next iteration
            v = vidList[i]
            nameAP = workD.append('chunks').append(v.path().split('.')[0] + '.mp3').aPath()
            grab = None
            if not workD.append('chunks').append(v.path().split('.')[0] + '.mp3').exists():
                ffmpeg.input(v.aPath()).filter("afftdn", nr=6, nt="w", om="o").output(nameAP).run(overwrite_output=True)
            pv = v.getProcessedVideo()
            if pv.duration > DEFAULT_MAX_CHUNK_SIZE:
                grab = v.chunk()
            else:
                grab = [pv]
            #for item in grab:
            audioProcess = AudioSegment.from_mp3(nameAP)
            chunksProcess = make_chunks(audioProcess, chuLenMS)
            iterations = math.floor(pv.duration / chuLenS)
            for i in range(iterations - 1):
                ts = i * chuLenS
                tf = (i + 1) * chuLenS
                videoChunks.append(kChunk(pv.subclip(ts, tf), ts, tf, chunksProcess[i].dBFS, nameAP))
            chunksProcess = list(map(lambda x: self.floor_out(x.dBFS, -300), chunksProcess))
            apList += chunksProcess
            del pv
            self.x += 50 // length
            self.progress()
        #...and normalize
        #apList = list(map(lambda x: x + 300, chunksProcess))
        #print(self.normalize(apList, self.defaultGet))
        #print(list(filter(lambda x: 1 < x or 0 > x, (self.normalize(apList, self.defaultGet) + 1) / 2)))
        #apNorm = (apList - np.mean(apList)) / np.ptp(apList)
        apNorm = (self.normalize(apList, self.defaultGet) + 1) / 2
        finalClip = []
        print(f'len(videoChunks) = {len(videoChunks)}, len(apNorm) = {len(apNorm)}')
        length = len(videoChunks)
        for i in range(length - 1):
            if apNorm[i] >= DEFAULT_THRESHOLD or self.computeSV(videoChunks, i) >= DEFAULT_REACH_THRESH:
                finalClip.append(videoChunks[i])
            self.x += 50 // length
            self.progress()
        print(f'len(finalClip) = {len(finalClip)}')
        finalClip = list(map(lambda d: d.content, finalClip))
        self.endProgress()
        print(finalClip)
        outputMovie = mpye.concatenate_videoclips(finalClip)
        outputMovie.write_videofile(outD.append('output.mp4').aPath(), codec='libx265')


    def startProgress(self):
        sys.stdout.write(self.title + ": [" + "-"*40 + "]" + chr(8)*41)
        sys.stdout.flush()
        self.progress_x = 0

    def progress(self):
        x = int(self.x * 40 // 100)
        sys.stdout.write("#" * (x - self.progress_x))
        sys.stdout.flush()
        self.progress_x = x

    def endProgress(self):
        sys.stdout.write("#" * (40 - self.progress_x) + "]\n")
        sys.stdout.flush()


    def kSum(self, list, func):
        s = 0
        for o in list:
            s += func(o)
        return s


    def computeSV(self, list, i):
        minimum = max(0, i - (DEFAULT_REACH_ITER // 2))
        maximum = min(len(list) - 1, i + (DEFAULT_REACH_ITER // 2))
        s = self.kSum(list[minimum:maximum], self.getV) / max(1, maximum - minimum)
        return s


    def vidList(self, k):
        fSet = list(os.listdir(k.aPath()))
        vids = sorted(list(filter(lambda v: v[-4:].lower() in extList, fSet)))
        vids = list(map(lambda v: k.append(v), vids))
        return vids


    def defaultGet(self, element):
        return element


    def getV(self, element):
        return element.volume


    def getDBFS(self, element):
        return element.dBFS


    def floor_out(self, a, bottom):
        if a < bottom:
            return bottom
        else:
            return a


    def normalize(self, list, fn):
        values = []
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
    outD = workD.append('output')
    kss2(sessID, inD, workD, outD)
