#  TODO:  /
#      -  function that passes chunks to main class
#         /
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

import math
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

import json

#time

from datetime import date

#import tensorflow as tf
#import numpy as np

# classes

#kPath handles a lot of basic file path issues and makes coding file directories a breeze
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
        if self.path()[-4:].lower() in extList:
            list = []
            video = mpye.VideoFileClip(self.p)
            d = video.duration
            if d < 0:
                return None
            n = int(d // DEFAULT_MAX_CHUNK_SIZE)
            for i in range(0, n - 1):
                list.append(video.subclip(i * DEFAULT_MAX_CHUNK_SIZE, (i + 1) * DEFAULT_MAX_CHUNK_SIZE))
            return list
        else:
            return None


    def getProcessedVideo(self):
        clips = list()
        k = mpye.VideoFileClip(self.p)
        if k.duration > DEFAULT_MAX_CHUNK_SIZE:
            clips = self.chunk()
        else:
            clips.append(k)

        return k

        if len(clips) is 1:
            return k
        elif len(clips) is 0 and clips is not None:
            return k
        else:
            return None

#kChunk handles storing audio data
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

#kss provides the process for making the edits
class kss:

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
        self.x = 100 #set max length of progress bar
        self.progress_x = 0
        self.title = 'chunking'
        #self.startProgress()
        length = len(vidList)
        for i in range(length): #now make a list of kChunks so that the program can sticth video and audio in the next iteration
            v = vidList[i]
            print(f"video #{i} = {v}")
            nameAP = workD.append('chunks').append(v.path().split('.')[0] + '.mp3').aPath()
            if not workD.append('chunks').append(v.path().split('.')[0] + '.mp3').exists():
                ffmpeg.input(v.aPath()).filter("afftdn", nr=6, nt="w", om="o").output(nameAP).run(overwrite_output=True)
            pv = v.getProcessedVideo()
            audioProcess = AudioSegment.from_mp3(nameAP)
            chunksProcess = make_chunks(audioProcess, chuLenMS)
            iterations = math.floor(pv.duration / chuLenS) + 1
            for i in range(len(chunksProcess)):
                if i % 50 == 0:
                    print(f"creating chunk #{i}")
                ts = i * chuLenS
                tf = (i + 1) * chuLenS
                if (tf > pv.duration):
                    tf = pv.duration
                videoChunks.append(kChunk(pv.subclip(ts, tf), ts, tf, chunksProcess[i].dBFS, nameAP))
            chunksProcess = list(map(lambda x: self.floor_out(x.dBFS, -300) + 300, chunksProcess))
            apList += chunksProcess
            del pv
            self.x += 50 // length
        max = 0
        #top_10 = sorted(apList, reverse=True)[:min(25, len(apList))]
        #if len(top_10) > 0:
        #    max = sum(top_10) / (len(top_10))
        #else:
        #    max = top_10
        for i in range(len(apList)): #normalize data
            if max < apList[i]:
                max = apList[i]
        for i in range (len(apList)):
            apList[i] /= (0.9 * 300) + (0.1 * max) # max
        finalClip = [] #build final clip
        print(f"building final clip")
        for i in range(len(apList)):
            if i % 10 == 0:
                print(f"filtering chunk #{i}")
            if apList[i] >= DEFAULT_THRESHOLD or self.computeSV(apList, i) >= DEFAULT_REACH_THRESH:
                print(f"  kept")
                finalClip.append(videoChunks[i])
            self.x += 50 // length
            #self.progress()
        finalClip = list(map(lambda d: d.content, finalClip))
        #self.endProgress()
        print('\r\n')
        outputMovie = mpye.concatenate_videoclips(finalClip)
        #labeling options
        tag = ''
        if include_program_tag:
            tag += f'[{program_name + program_version}] '
            if include_render_date:
                tag += f'(date={date.today()}) '
                if include_preset_name_in_output:
                    tag += f'(preset={preset_name}) '
        #rough draft for preview
        outputMovie.write_videofile(outD.append(tag + 'preview.mp4').aPath(), \
            codec = 'libx264', audio_codec='libmp3lame', audio_bitrate='22k', preset='veryfast', \
            threads=4)
        #final video
        outputMovie.write_videofile(outD.append(tag + 'output.mp4').aPath(), \
            codec = 'libx264', audio_codec='libmp3lame', audio_bitrate='96k', preset='veryslow', \
            threads=4)


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


    def kChunkComputeSV(self, list, i):
        minimum = max(0, i - (DEFAULT_REACH_ITER // 2))
        maximum = min(len(list) - 1, i + (DEFAULT_REACH_ITER // 2))
        total = 0
        for o in list[minimum:maximum]:
            total += o.volume
        ###s = sum() / max(1, maximum - minimum)
        return total / max(1, maximum - minimum)
        ###self.kSum(list[minimum:maximum], self.getV)


    def computeSV(self, list, i):
        minimum = max(0, i - (DEFAULT_REACH_ITER // 2))
        maximum = min(len(list) - 1, i + (DEFAULT_REACH_ITER // 2))
        return sum(list[minimum:maximum]) / max(1, maximum - minimum)


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

#functions outside of class scope


# id / random string generator
def randomString(stringLength=10):
    string = 'abcdefghijklmnopqrstuvwxyz1234567890!@&_'
    """Generate a random string of fixed length """
    letters = string
    return ''.join(random.choice(letters) for i in range(stringLength))

#col text box generation
def bordered(text):
    lines = text.splitlines()
    width = max(len(s) for s in lines)
    res = ['┌' + '─' * width + '┐']
    for s in lines:
        res.append('│' + (s + ' ' * width)[:width] + '│')
    res.append('└' + '─' * width + '┘')
    return '\n'.join(res)

#init
FFMPEG_BIN = 'ffmpeg'
dir = ''
dir = dirname(abspath(__file__)) + "\\footage"
chunks = []
gen = []

file = open('options.json', 'r')
data = json.load(file)

#program_info
program_name = data['program_info']['program_name']
program_version = data['program_info']['version']

#file_naming_options
include_program_tag = data['file_naming_options']['include_program_tag']
include_render_date = data['file_naming_options']["include_render_date"]
include_preset_name_in_output = data['file_naming_options']['include_preset_name_in_output']

#processing_options
default_input_folder = data['file_management_options']['default_input_folder']
cleanup = data['file_management_options']["clean_workspace_afterwards"]
extList = data['file_management_options']['find_videos_with_formats']


#processing_options
preset_name = data['processing_options']['selected_mode']
sp = data['processing_options']["modes"][preset_name]
DEFAULT_THRESHOLD = sp[0]
DEFAULT_PERIOD = sp[1]
DEFAULT_REACH_ITER = sp[2]
DEFAULT_REACH_THRESH = sp[3] * DEFAULT_THRESHOLD
DEFAULT_WIDTH = sp[4]  # 2560
DEFAULT_HEIGHT = sp[5]  # 1440
DEFAULT_MAX_CHUNK_SIZE = sp[6] * 60 #1.2, 3.2, 10.2
DEFAULT_TREATMENT = sp[7]
#list(['voice', 'music'])[0]
dir = ''

#console_settings
verbose = data['console_settings']["verbose"]
print_color = "cyan" #['console_settings']['default_print_color']
print_color_error = "yellow" #['console_settings']['default_error_color']

#print('\n=========\n\nsSafe Exit\n\n=========\n')
#exit()
sessID = randomString(4)
workD = kPath(dirname(abspath(__file__)) + "\\footage")
inD = data['file_management_options']['default_input_folder']
if '[workD]' in inD:
    inD = inD.replace('[workD]\\', '').replace('[workD]', '')
    inD = workD.append(inD)
outD = workD.append('output')

print(bordered(f'{program_name} version {program_version}'))

kss(sessID, inD, workD, outD)
