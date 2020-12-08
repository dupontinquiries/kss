"""
kCombine 1.0
Quick Video Packager
"""

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

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_audio

import json

import cv2
from pydub import AudioSegment
from pydub.utils import make_chunks
import math
import sys

#time

from datetime import date
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
        if self.p[-4:] in [ '.mp4', '.mov']:
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


def compressDir(root, inD, outD, fs, completedConversions, failedConversions, subF):
    """
    root = hard path of parent folder
    inD  = directory of videos to compress; changes with recursion
    outD = output folder to deliver to; constant with recursion
    fs   = file size limit to deliver to ffmpeg processing
    """
    #print('inD = {0}'.format(inD))
    #print('outD = {0}'.format(outD))
    for name in os.listdir(inD.aPath()):
        #print('inD = {0}'.format(inD.aPath()))
        fPath = kPath(inD.append(name))
        #print('{0}, {1}, {2}'.format(root.aPath(), inD.aPath(), outD.aPath()))
        #print('{0}, {1}, {2}'.format(nameRoot, nameExt, fPath.aPath()))
        if fPath.isFile():
            nameRoot = name[:-4]
            nameExt = name[-4:]
            if nameExt in ['.mp3', '.wav', '.zip']:
                continue
            if nameExt.lower() not in ['.mp4', '.mkv']:
                print(name)
                continue
            if fPath.aPath() in completedConversions:
                continue
            out = outD
            if subF != '':
                out = out.append(subF)
            if not out.exists():
                os.mkdir(out.aPath())
            out = out.append(nameRoot)
            out = out.hitch('_kCrawler.mp4')
            if out.exists():
                continue
            cmd = ( f'ffmpeg -y -i "{fPath}" -c:v libx265 -crf 19'
                +   f' -level 3.1 -preset slow'
                +   f' -sws_flags lanczos -c:a aac -b:a 192k -vbr 5 "{out}"' )
            #cmd = f'ffmpeg -y -i "{fPath}" -c:v libx265 -c:a aac "{out}"'
            #print(outD)
            #print(cmd)
            os.system('{0}'.format(cmd))
            completedConversions.add(fPath.aPath())
        else:
            subFB = subF
            if subF != '':
                subF = subF + '\\'
            subF = subF + name
            if not outD.append(subF).exists():
                if inD.append(subF).isFolder():
                    outD.append(subF).make()
            #print('subF was ({0}) now ({1})'.format(subFB, subF))
            compressDir(root, inD.append(name), outD, fs, completedConversions, failedConversions, subF) #add path changes here as a string and append it to the outD
            subF = ''

# essentials

def vidList(k):
    fSet = list(os.listdir(k.aPath()))
    vids = sorted(list(filter(lambda v: v[-4:].lower() in ['.mp4', '.mkv', '.mov'], fSet)))
    vids = list(map(lambda v: k.append(v), vids))
    return vids

def linePrint( v ):
    r = ''
    for o in v[:-1]:
        r += o.path() + '\n'
    r += v[-1].path()
    return r

# id / random string generator
def randomString(stringLength=10):
    string = 'abcdefghijklmnopqrstuvwxyz1234567890!@&_'
    """Generate a random string of fixed length """
    letters = string
    return ''.join(random.choice(letters) for i in range(stringLength))

# main

root = 'M:\\2019\\Recordings 2019\\GoPro\\2019-11-09\\HERO8 BLACK 1'
root = kPath(kIn('Path to the workspace => '))

files = vidList( root )

default = True
if default:
    files = sorted( files, key = lambda x: x.path() )
else:
    for i in range( len( files ) ):
        print("{0:>3}: {1}".format( i, files[i].path() ) )
    o = list( kIn('Please select your custom sort order:\n     ').split(', ') )
    tmp = files.copy()
    for i in range( len( o ) ):
        files[i] = tmp[ int(o[i]) ]

for i in range( len( files ) ):
    print("{0:>3}: {1}".format( i, files[i].path() ) )
#print( '{0}'.format( linePrint( files ) ) )

os.chdir(root.aPath())

completedConversions = set()
failedConversions = set()

subclips = list()
for f in files:
    subclips.append( mpye.VideoFileClip( f.path() ) )

out = mpye.concatenate_videoclips(subclips, method='compose')
output = root.append(f'{randomString(5)}_compilation.mp4')
out.write_videofile(output.aPath(), preset='ultrafast', codec='libx265', audio_codec='aac', audio_bitrate='48k')


#compressDir(root, root, outD, fs, set(), set(), '')
