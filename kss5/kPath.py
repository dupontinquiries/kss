"""
KSS 5.0
Kitchen Silence Splitter
"""

#imports

from __future__ import unicode_literals

#av

#video

import ffmpeg
from moviepy.video.io.ffmpeg_tools import *
import moviepy
import moviepy.editor as mpye
from moviepy.editor import *

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

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio

import json

import cv2
from pydub import AudioSegment
from pydub.utils import make_chunks
import math
import sys

#time

from datetime import date

#import tensorflow as tf
#import numpy as np

import multiprocessing
import threading
import concurrent

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
        return k


        if k.duration > DEFAULT_MAX_CHUNK_SIZE:
            clips = self.chunk()
        else:
            return (False, None, k)
        #if len(clips) > 1: #as array
        if type(clips) is list:
            return (True, clips, k)
        else:
            return (False, None, k)
        #elif len(clips) is 1: #as single clip
        #    return k[0]
        #elif len(clips) is 0 and clips is not None: #as single clip
        #    return k
        #else:
        #    return None
