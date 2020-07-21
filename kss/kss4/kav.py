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

#from termcolor import colored

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


    def delete(self):
        if os.path.exists(self.p) and self.p[-4] == '.':
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
        if self.p[-4:] in extList:
            result = subprocess.Popen(["ffprobe", self.p], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            return [x for x in result.stdout.readlines() if "Duration" in x]
        else:
            return -1


    def chunk(self): #does no checking
        chunks = list()
        video = mpye.VideoFileClip(self.p)
        d = video.duration
        n = int(d) // DEFAULT_MAX_CHUNK_SIZE + 1
        mt = DEFAULT_MAX_CHUNK_SIZE
        for i in range(n):
            time = (i * mt, (i + 1) * mt)
            if time[1] > d:
                subclip = video.subclip(i * mt, d)
                chunks.append(subclip)
                break
            subclip = video.subclip(time[0], time[1])
            chunks.append(subclip)
        return chunks


    def getFullVideo(self):
        return mpye.VideoFileClip(self.p)

    def getProcessedVideo(self):
        clips = list()
        k = mpye.VideoFileClip(self.p)
        if k.duration > DEFAULT_MAX_CHUNK_SIZE:
            clips = self.chunk()
        else:
            return (False, None, k)
        print(f"getPV returned: {clips}")
        #if len(clips) > 1: #as array
        if type(clips) is list:
            print("this return")
            return (True, clips, k)
        else:
            return (False, None, k)
        #elif len(clips) is 1: #as single clip
        #    return k[0]
        #elif len(clips) is 0 and clips is not None: #as single clip
        #    return k
        #else:
        #    return None

#kChunk handles storing audio data
class kChunk:

    data = []

    def __init__(self, content, ts, tf, volume, sourceName):
        self.content = content
        self.ts = ts
        self.tf = tf
        self.timestamp = (self.ts, self.tf)
        self.volume = volume
        self.sv = None
        self.sourceName = sourceName
        self.data += [self.content, self.ts, self.tf, self.volume, self.sourceName]


    def __repr__(self):
        return repr('[CHUNK] @ {0}, v = {1:.3f}, sv = {2:.3f}'.format(self.timestamp, self.volume, self.sv))


    def __eq__(self, b):
        return self.t_s == b.t_s and self.t_f == b.t_f and self.sourceName == b.sourceName


    def floor_out(self, bottom):
        if self.volume < bottom:
            return bottom
        self.volume -= bottom
        return self

#kss provides the process for making the edits
class kss:

    def extractAudio(self, i, vidList):
        v = vidList[i]
        nameAP = workD.append('chunks').append(v.path().split('.')[0] + '.mp3').aPath()
        if not workD.append('chunks').append(v.path().split('.')[0] + '.mp3').exists():
            if DEFAULT_TREATMENT == 'game':
                (
                ffmpeg.input(v.aPath())
                    .filter('lowpass', f=18000).filter('highpass', f=20) .filter('extrastereo', m=1.3)
                    .filter('equalizer', f=20, t='q', w=0.5, g=0.4)
                    .filter('equalizer', f=80, t='q', w=0.5, g=0.1)
                    .filter('equalizer', f=20000, t='q', w=0.3, g=0.2)
                    .filter('dynaudionorm', f=630, g=201, m=0.5)
                    .filter('afftdn', nr=1, nt="w", om="o")
                    .output(nameAP).run(overwrite_output=True)
                )
            if DEFAULT_TREATMENT == 'music':
                (
                ffmpeg.input(v.aPath())
                    .filter('lowpass', f=20000).filter('highpass', f=20)
                    .filter('afftdn', nr=1, nt="w", om="o")
                    .output(nameAP).run(overwrite_output=True)
                )
            if DEFAULT_TREATMENT == 'voice':
                (
                ffmpeg.input(v.aPath())
                    .filter('lowpass', f=18000).filter('highpass', f=20)
                    .filter('deesser').filter('dynaudionorm', f=230, g=71, m=3)
                    .filter('afftdn', nr=1, nt="w", om="o")
                    .output(nameAP).run(overwrite_output=True)
                )
            if DEFAULT_TREATMENT == 'focusedVoice':
                (
                ffmpeg.input(v.aPath())
                    .filter('lowpass', f=17000).filter('highpass', f=40)
                    .filter('deesser')
                    .filter('dynaudionorm', f=230, g=71, m=3)
                    .filter('afftdn', nr=1, nt="w", om="o")
                    .output(nameAP).run(overwrite_output=True)
                )
            if DEFAULT_TREATMENT == 'noisy':
                (
                ffmpeg.input(v.aPath())
                    .filter('lowpass', f=18000).filter('highpass', f=80)
                    .filter('afftdn', nr=4, nt="w", om="o")
                    .filter('dynaudionorm', f=330, g=101, m=1)
                    .filter('afftdn', nr=2, nt="w", om="o")
                    .filter('dynaudionorm', f=330, g=101, m=0.5)
                    .filter('afftdn', nr=1, nt="w", om="o")
                    .output(nameAP).run(overwrite_output=True)
                )
        return v

    def chunkAudio(self, v):
        pvc = v.getFullVideo()
        # create tandem mp3 audio
        af = workD.append("footage").append("chunks").append(f"chunk_{randomString(7)}.mp3")
        AudioFileClip(af.aPath())
        a = AudioSegment.from_mp3(nameAP)
        packets = make_chunks(a, chuLenMS)
        # make 5 minute segments to process simultaneously
        n = pvc.duration // 300
        subclips = list()
        for i in range(n):
            ts = n * 300
            tf = (n + 1) * 300
            tf = max(tf, pvc.duration)
            subclips.append(pvc.subclip(ts, tf))

        self.tmpChunks = list()
        self.tmpCounter = 0
        executor = concurrent.futures.ProcessPoolExecutor(61)
        futures = [executor.submit(self.appendChunks, subclips[i], i)
                   for i in range(len(subclips))]
        # run code in the meantime
        concurrent.futures.wait(futures)
        # order in case concurrent was out of order
        self.tmpChunks = sorted(self.tmpChunks, key=lambda element: (element[0], element[1]))
        for i in range(len(self.tmpChunks)):
            i1 = max(0, i - spreadCalc)
            i2 = min(len(self.tmpChunks), i + spreadCalc)
            self.tmpChunks[i].sv = sum(list(map(lambda x: x.volume, self.tmpChunks[i1:i2])) / max(1, i2 - i1))
        print(self.tmpChunks)
        os.remove(af.aPath())
        exit()


    def appendChunks(self, subclip):
        self.tmpCounter += 1
        chunks = list()
        n = subclip.duration // chuLenS
        for i in range(n):
            ts = n * chuLenS
            tf = (n + 1) * chuLenS
            tf = max (tf, pvc.duration)
            self.tmpChunks.append((self.tmpCounter, i, kChunk(subclip.subclip(ts, tf), ts, tf, chunksProcess[totalChunks].dBFS, nameAP)))


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
        spreadCalc = DEFAULT_REACH_ITER
        chuLenMS = DEFAULT_PERIOD
        chuLenS = chuLenMS / 1000
        apList = list()
        videoChunks = list()
        tmpVideoChunks = list()
        length = len(vidList)
        import concurrent
        import threading
        executor = concurrent.futures.ProcessPoolExecutor(61)
        futures = [executor.submit(self.extractAudio, i, vidList)
                   for i in range(length)]
        # run code in the meantime
        concurrent.futures.wait(futures)
        # run code once preprocessed audio is ready

        # create chunk lists
        self.chunkList = list()
        executor = concurrent.futures.ProcessPoolExecutor(61)
        futures = [executor.submit(self.chunkAudio, vidList[i])
                   for i in range(length)]
        # run code in the meantime
        concurrent.futures.wait(futures)
        # run code once chunks are ready
        print(f"vidList [s={len(vidList)}] = {vidList}")
        exit()
        for i in range(length): #now make a list of kChunks so that the program can sticth video and audio in the next iteration
            pvBool, pvList, pvc = v.getProcessedVideo()
            if pvList == None:
                print("None encountered")
            if pvBool == True:
                print("a: " + str(pvBool))
                print("b: " + str(pvList))
                print("c: " + str(pvc))
                print("\n")
                pvc.audio = mpye.AudioFileClip(nameAP)
                audioProcess = AudioSegment.from_mp3(nameAP)
                chunksProcess = make_chunks(audioProcess, chuLenMS)
                iterations = math.floor(pvc.duration / chuLenS)
                totalChunks = 0
                for scn in range(iterations): #subclip in pv:
                    subclip = pvList[scn]
                    #nameAP = workD.append('chunks').append(v.path().split('.')[0] + 'chunk' + i + '.mp3').aPath()
                    dur = subclip.duration
                    print(f"dur = {dur}")
                    #videoChunks = list(map(lambda i: , chunksProcess))
                    tmpVideoChunks = list()
                    lcp = len(chunksProcess) - 1
                    for i in range(lcp):
                        ts = i * chuLenS
                        tf = (i + 1) * chuLenS
                        #ts += scn * DEFAULT_MAX_CHUNK_SIZE
                        #tf += scn * DEFAULT_MAX_CHUNK_SIZE
                        if totalChunks > lcp:
                            break
                        if tf >= dur or ts >= dur:
                            break
                        tmpVideoChunks.append(kChunk(subclip.subclip(ts, tf), ts, tf, chunksProcess[totalChunks].dBFS, nameAP))
                        totalChunks += 1
                    try:
                        tmpVideoChunks = list(map(lambda x: x.floor_out(-500), tmpVideoChunks))
                        apList += tmpVideoChunks
                    except ValueError:
                        break
                    del subclip
            else: #single clip
                pvc.audio = mpye.AudioFileClip(nameAP)
                audioProcess = AudioSegment.from_mp3(nameAP)
                chunksProcess = make_chunks(audioProcess, chuLenMS)
                dur = pvc.duration
                iterations = math.floor(dur / chuLenS)
                for i in range(iterations):
                    ts = i * chuLenS
                    tf = (i + 1) * chuLenS
                    if tf >= dur or ts >= dur:
                        break
                    try:
                        tmpVideoChunks.append(kChunk(pvc.subclip(ts, tf), ts, tf, chunksProcess[i].dBFS, nameAP))
                    except ValueError:
                        break
                for ch in tmpVideoChunks:
                    ch = ch.floor_out(-500)
                #tmpVideoChunks = list(map(lambda x: x.floor_out(-500), tmpVideoChunks))
                apList += tmpVideoChunks
                del pvc
        max = 0
        #for i in range(len(apList)): #find rel max in data
        #    if max < apList[i]:
        #        max = apList[i]
        for i in range (len(apList)):
            apList[i].volume /= (1.0 * 500) # + (0.1 * max) # max
        finalClip = list() #build final clip
        for i in range(len(apList)):
            #print(f"vol = {apList[i].volume}\n sv = {self.computeSV(apList, i)}")
            if apList[i].volume >= DEFAULT_THRESHOLD or self.computeSV(apList, i) >= DEFAULT_REACH_THRESH:
                finalClip.append(videoChunks[i])
        finalClip = list(map(lambda d: d.content, finalClip))
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
        #outputMovie.write_videofile(outD.append(tag + 'preview.mp4').aPath(), \
        #    codec='libx265', audio_codec='libmp3lame', \
        #    audio_bitrate='48k', preset='veryfast', \
        #    threads=4)
        #final video
        outputMovie.write_videofile(outD.append(tag + 'output.mp4').aPath(), \
            codec='libx265', audio_codec='libmp3lame', \
            audio_bitrate='96k', preset='fast', \
            threads=16)
        if cleanup:
            fSet = list(os.listdir(k.aPath()))
            for file in fSet:
                file.delete()


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
        subList = list[minimum:maximum]
        sum = 0
        for o in subList:
            sum += o.volume
        sum /= max(1, maximum - minimum)
        return sum

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
else:
    inD = kPath(inD)
outD = workD.append('output')

print(bordered(f'{program_name} version {program_version}'))

kss(sessID, inD, workD, outD)
