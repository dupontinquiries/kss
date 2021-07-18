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
        v = kPath('/'.join(self.p.split('/')[:-1]))
        return kPath(v)


    def cascadeCreate(self, p):
        pChunks = p.split('/')
        s = pChunks[0]
        end = len(pChunks)
        for i in range(1, end):
            s += '/' + pChunks[i]
            if s[-4] == '.' or 'mp4' in p:
                continue
            elif not os.path.exists(s):
                os.mkdir(s)


    def append(self, w):
        v = self.p + '/' + w
        return kPath(v)


    def make(self):
        os.mkdir(self.p)


    def hitch(self, w):
        v = self.p + w
        return kPath(v)


    def path(self):
        return self.p.split('/')[-1]


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

#kChunk handles storing audio data
class kChunk:

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

#kChunk handles storing audio data
class kChunk:

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

    def __init__(self):
        print('new instance')

    def extractAudio(self, i, vidList):
        v = vidList[i]
        nameAP = workD.append('chunks').append(v.path().split('.')[0] + '.mp3').aPath()
        if not workD.append('chunks').append(v.path().split('.')[0] + '.mp3').exists():
            if DEFAULT_TREATMENT == 'none':
                ffmpeg.input(v.aPath()).output(nameAP).run(overwrite_output=True)
            if DEFAULT_TREATMENT == 'game':
                (
                ffmpeg.input(v.aPath())
                    .filter('lowpass', f=18000).filter('highpass', f=20) .filter('extrastereo', m=1.3)
                    .filter('equalizer', f=20, t='q', w=0.5, g=0.4)
                    .filter('equalizer', f=80, t='q', w=0.5, g=0.1)
                    .filter('equalizer', f=20000, t='q', w=0.3, g=0.2)
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
                    .filter('afftdn', nr=1, nt="w", om="o")
                    .output(nameAP).run(overwrite_output=True)
                )
            if DEFAULT_TREATMENT == 'focusedVoice':
                (
                ffmpeg.input(v.aPath())
                    .filter('lowpass', f=17000).filter('highpass', f=40)
                    .filter('deesser')
                    .filter('afftdn', nr=1, nt="w", om="o")
                    .output(nameAP).run(overwrite_output=True)
                )
            if DEFAULT_TREATMENT == 'noisy':
                (
                ffmpeg.input(v.aPath())
                    .filter('lowpass', f=18000).filter('highpass', f=80)
                    .filter('afftdn', nr=4, nt="w", om="o")
                    .filter('afftdn', nr=2, nt="w", om="o")
                    .filter('afftdn', nr=1, nt="w", om="o")
                    .output(nameAP).run(overwrite_output=True)
                )
            if DEFAULT_TREATMENT == 'noisy2':
                (
                ffmpeg.input(v.aPath())
                    .filter('lowpass', f=18000).filter('highpass', f=80)
                    .filter('deesser')
                    .filter('afftdn', nr=4, nt="w", om="o")
                    .output(nameAP).run(overwrite_output=True)
                )
        return v

    def testHello(self):
        print("hello func")

    def chunkAudio_createChunk(self, i, pvc, subclips):
        ts = i * self.chuLenS
        tf = (i + 1) * self.chuLenS
        tf = max(tf, pvc.duration)
        subclips.append(pvc.subclip(ts, tf))

    def chunkAudio(self, v):
        print(f"chunkAudio session {randomString(4)}")
        pvc = v.getFullVideo()
        # create tandem mp3 audio
        af = self.workD.append("chunks").append(f"chunk_{randomString(7)}.mp3")
        #print(f"created temporary file {af.path()}")
        ffmpeg_extract_audio(v.aPath(), af.aPath()) #, ffmpeg_params=["-preset","fast"])
        a = AudioSegment.from_mp3(af.aPath())
        print(a)
        packets = make_chunks(a, self.chulenms)
        print(f"dividing clip")
        # make 5 minute segments to process simultaneously
        n = int(pvc.duration // self.chuLenS)

        import concurrent.futures

        from multiprocessing import Pool
        #with multiprocessing.Pool() as p:
        #    p.map(self.chunkAudio_createChunk, range(n))

        subclips = [None] * n #list()
        executor = concurrent.futures.ProcessPoolExecutor(61)
        futures = [executor.submit(self.chunkAudio_createChunk, i, pvc, subclips)
                   for i in range(n)]

        #for i in range(n):
        print(f"   breakpoint 1")
        #self.tmpChunks = list() # cannot perform
        #self.tmpCounter = 0
        print(f"preparing jobs for list of size {len(subclips)}")

        self.tmpChunks = list()
        executor = concurrent.futures.ProcessPoolExecutor(61)
        futures = [executor.submit(self.appendChunks, subclips[i], i, self.tmpChunks)
                   for i in range(len(subclips))]
        # run code in the meantime
        concurrent.futures.wait(futures)
        print(f"aggregated all chunks")

        ######
        # order in case concurrent was out of order
        self.tmpChunks = sorted(self.tmpChunks, key=lambda element: (element[0], element[1]))
        print(f"organized all chunks")
        for i in range(len(self.tmpChunks)):
            i1 = max(0, i - spreadCalc)
            i2 = min(len(self.tmpChunks), i + spreadCalc)
            self.tmpChunks[i].sv = sum(list(map(lambda x: x.volume, self.tmpChunks[i1:i2])) / max(1, i2 - i1))
        print(f"spread volumes calculated")
        print(self.tmpChunks)
        af.delete() #os.remove(af.aPath())
        print(f"destroyed temporary file {af.path()}")
        ######exit()


    def appendChunks(self, subclip, givenCounter, retList):
        #self.tmpCounter += 1
        #chunks = list()
        n = subclip.duration // self.chuLenS
        for i in range(n):
            ts = n * chuLenS
            tf = (n + 1) * chuLenS
            tf = max (tf, pvc.duration)
            retList.append( ( givenCounter, i, kChunk( subclip.subclip(ts, tf), ts, tf, chunksProcess[totalChunks].dBFS, nameAP ) ) )


    def makeAudio(self, i, vidList, audioFileList, UUID):
        print(f"({UUID}) makeAudio()")
        #v = self.extractAudio(i, vidList)
        #pvc = v.getFullVideo()
        # create tandem mp3 audio
        af = self.workD.append("chunks").append(f"chunk_{randomString(7)}.mp3")
        #print(f"created temporary file {af.path()}")
        ffmpeg_extract_audio(vidList[i].aPath(), af.aPath())
        audioFileList.append(af)
        print(f"  end ({UUID}) makeAudio()")


    def convertMP3toChunksHelper(self, i, packets):
        packets[i] = ( *(packets[i]), i * self.chulenms, (i + 1) * self.chulenms )

    def convertMP3toChunksGrouper(self, i, g, packets):
        for i in range( i, min( g, len(packets) )):
            self.convertMP3toChunksHelper(i, packets)

    def convertMP3toChunks(self, i, audioFileList, audioChunksList, UUID):
        print(f"({UUID}) convertMP3toChunks()")
        a = AudioSegment.from_mp3( audioFileList[i].aPath() )
        packets = make_chunks(a, self.chulenms)
        packets = list( map(lambda x: (i, x, x.dBFS), packets) )

        switch = False
        if switch:
            executor = concurrent.futures.ProcessPoolExecutor(16)
            futures = [executor.submit(self.convertMP3toChunksGrouper, i, 100, packets)
               for i in range( 0, len(packets), 100 )]
            concurrent.futures.wait(futures)

        else:
            for j in range( len(packets) ):
                self.convertMP3toChunksHelper(j, packets)
            #for i in range( len(packets) ):
            #    packets[i] = ( *(packets[i]), i * self.chulenms, (i + 1) * self.chulenms )

        audioChunksList += packets
        audioFileList[i].delete()
        print(f"  end ({UUID}) convertMP3toChunks()")


    def convertMP3toChunksOrganizer(self, group, audioFileList, audioChunksList, UUID):
        for i in group:
            self.convertMP3toChunks(i, audioFileList, audioChunksList, UUID)


    def helperFloorChunk(self, i, audioChunksList, bottom):
        if audioChunksList[i][2] < bottom:
            audioChunksList[i][2] = bottom


    def floorChunks(self, i, audioChunksList):
        ###print(f"(main) floorChunks()")
        #executor = concurrent.futures.ProcessPoolExecutor(4)
        #futures = [executor.submit(self.helperFloorChunk, i, audioChunksList, -500)
        #for i in range(len(audioChunksList))]
        #run code in the meantime
        #concurrent.futures.wait(futures)
        #for i in range(len(audioChunksList)):
        self.helperFloorChunk( i, audioChunksList, -500 )

    def helperCalculateSV(self, i, audioChunksList, audioDataList, bottom):
        print(f"calculating spread value: {i + 1}/{len(audioChunksList)}\r", end="")
        a = max(0, i - 5)
        b = min(len(audioChunksList), i + 5)
        tmp = list(map(lambda x: x[2], audioChunksList[a:b])) # audioChunksList[a:b]
        #print(f"{tmp}\n{sum(tmp)}\n{max(1, b - a)}\n{sum(tmp) / max(1, b - a)}\n\n{(audioChunksList[i])}")
        if b - a < 1:
            #print(f"{audioChunksList[i]}")
            #print(f"{tmp}")
            #print(f"{b - a}")
            #print(f"{audioChunksList[i]}")
            #exit()
            audioDataList[i] = ( *(audioChunksList[i]), tmp )
        else:
            #print(f" audioChunk: {audioChunksList[i]} \n tmp: {tmp}")
            #print(f"{audioChunksList[i]}")
            #exit()
            audioDataList[i] = ( *(audioChunksList[i]), sum(tmp) / max(1, b - a) )
            #audioDataList[i] = ( audioChunksList[i][0], audioChunksList[i][1], audioChunksList[i][2], sum(tmp) / max(1, b - a) )
        #print(f"{audioDataList[i]}")
        #exit()


    # unused
    def calculateSV(self, audioChunksList):
        #print(f"(main) floorChunks()")
        #executor = concurrent.futures.ProcessPoolExecutor(61)
        #futures = [executor.submit(self.helperCalculateSV, i, audioChunksList)
        #for i in range(len(audioChunksList))]
        #run code in the meantime
        #concurrent.futures.wait(futures)
        print(f"\rcalculating spread value: {0}/{len(len(audioChunksList))}")
        for i in range( len(audioChunksList) ):
            self.helperCalculateSV(i, audioChunksList)
        print(f"\r\n")

    def generateSubclips():
        return True


    def writeVideoGrouper(self, i, g, tmpChunks, returnChunksList, tmpName, folderName, referenceVideos, vidList):
        clamp = min( i + g, len(tmpChunks) - 1 )
        if i < clamp:
            print(f"({tmpName[:9]}) writeVideoGrouper {i} - {clamp}")
            if len(tmpChunks[i:clamp]) > 0:
                _v = list()
                k = i
                tmpMpyVideos = dict()
                while k < clamp:
                    if tmpChunks[k][0]:
                        j = k
                        while tmpChunks[k][0] and tmpChunks[k][1] == tmpChunks[j][1] and k < len(tmpChunks) - 1:
                            k += 1
                        video = None
                        #try:
                        if tmpChunks[k][1] not in tmpMpyVideos:
                            video = vidList[tmpChunks[k][1]].getFullVideo()
                            tmpMpyVideos[tmpChunks[k][1]] = video
                        else:
                            video = tmpMpyVideos[tmpChunks[k][1]]
                        #video = referenceVideos[ tmpChunks[k][1] ]
                        #except:
                        #    video = vidList[tmpChunks[k][1]].getFullVideo()
                        a = max(0, tmpChunks[j][4] / 1000)
                        b = min( video.duration, tmpChunks[k][5] / 1000 )
                        fps = video.fps
                        _v.append( video.subclip( min(a, b), max(a, b) ).set_fps(fps) )
                    k += 1
                if len(_v) == 0:
                    return
                outputMovie = None
                if len(_v) == 1:
                    outputMovie = _v
                elif len(_v) > 1:
                    outputMovie = mpye.concatenate_videoclips(_v, method='compose')
                clip = outD.append(f'{folderName}_threading').append(f'{i//g} - {tmpName}_tmpChunk.mp4')
                outputMovie.write_videofile(clip.aPath(), preset='slow', threads=16) #, codec='libx265', audio_codec='aac', audio_bitrate='48k', threads=16) #12
                returnChunksList.append((i, clip))


    def writeVideo_multithreading(self, tmpChunks, referenceVideos, vidList):
        print(f"begin (main) writeVideo()")
        import threading
        threads = list()
        returnChunksList = list()
        nConcurrentThreads=1+(len(tmpChunks)//self.chulenms//40)
        sparsity = len(tmpChunks)//nConcurrentThreads
        folderName = randomString(9)

        count = 0
        for i in range( 0, len(tmpChunks), sparsity ):

            if len(threads) > 10:
                threads[0].join()
                threads.pop(0)

            t = threading.Thread(target=self.writeVideoGrouper, args=(i, sparsity, tmpChunks, returnChunksList, randomString(9), folderName, referenceVideos, vidList, ))
            threads.append(t)
            t.start()
            count += 1

        for thread in threads:
            thread.join()

        videoList = list()
        for i, path in sorted(returnChunksList):
            videoList.append(path.getFullVideo())


        outputMovie = mpye.concatenate_videoclips(videoList)
        outputMovie.write_videofile(outD.append(f'{randomString(5)} -- output.mp4').aPath(), preset='slow', codec='libx265', audio_codec='aac', audio_bitrate='48k') #, threads=16)

        for i, path in sorted(returnChunksList):
            if False:
                path.delete()

        cleanup = False
        if cleanup:
            executor = concurrent.futures.ProcessPoolExecutor(6)
            futures = [executor.submit(tmpChunks[i].delete())
               for i in range( len(tmpChunks) )]
            concurrent.futures.wait(futures)
        print(f"  end (main) writeVideo()")


    def writeVideoGrouper_multiprocessing(self, queue, i, g, tmpChunks, tmpName, folderName, vidList):
        clamp = min( i + g, len(tmpChunks) - 1 )
        if i < clamp:
            print(f"({tmpName[:9]}) writeVideoGrouper {i} - {clamp}")
            if len(tmpChunks[i:clamp]) > 0:
                _v = list()
                k = i
                tmpMpyVideos = dict()
                while k < clamp:
                    if tmpChunks[k][0]:
                        j = k
                        while tmpChunks[k][0] and tmpChunks[k][1] == tmpChunks[j][1] and k < len(tmpChunks) - 1:
                            k += 1
                        video = None
                        #try:
                        if tmpChunks[k][1] not in tmpMpyVideos:
                            video = vidList[tmpChunks[k][1]].getFullVideo()
                            tmpMpyVideos[tmpChunks[k][1]] = video
                        else:
                            video = tmpMpyVideos[tmpChunks[k][1]]
                        a = max(0, tmpChunks[j][4] / 1000)
                        b = min( video.duration, tmpChunks[k][5] / 1000 )
                        fps = video.fps
                        _v.append( video.subclip( min(a, b), max(a, b) ).set_fps(fps) )
                    k += 1
                if len(_v) == 0:
                    return
                outputMovie = None
                if len(_v) == 1:
                    outputMovie = _v
                elif len(_v) > 1:
                    outputMovie = mpye.concatenate_videoclips(_v, method='compose')
                clip = outD.append(f'{folderName}_threading').append(f'{i//g} - {tmpName}_tmpChunk.mp4')
                outputMovie.write_videofile(clip.aPath(), preset='slow', threads=16)
                queue.put( (i, clip) )


    def writeVideo_multiprocessing(self, tmpChunks, vidList):
        print(f"begin (main) writeVideo()")
        from multiprocessing import Process, Queue
        queue = Queue()
        nConcurrentThreads=1+(len(tmpChunks)//self.chulenms//25)
        sparsity = len(tmpChunks)//nConcurrentThreads
        folderName = randomString(9)

        processes = [ Process(  target=self.writeVideoGrouper_multiprocessing, args=(queue, i, sparsity, tmpChunks, randomString(9), folderName, vidList, )  ) for i in range( 0, len(tmpChunks), sparsity ) ]

        count = 0
        for p in processes:
            p.start()
            ++count

        for p in processes:
            p.join()

        results = [queue.get() for p in processes]

        videoList = list()
        for i, path in sorted(results):
            videoList.append(path.getFullVideo())

        outputMovie = mpye.concatenate_videoclips(videoList)
        outputMovie.write_videofile(outD.append(f'{randomString(5)} -- output.mp4').aPath(), preset='slow', codec='libx265', audio_codec='aac', audio_bitrate='48k') #, threads=16)

        cleanup = False
        if cleanup:
            executor = concurrent.futures.ProcessPoolExecutor(6)
            futures = [executor.submit(tmpChunks[i].delete())
               for i in range( len(tmpChunks) )]
            concurrent.futures.wait(futures)
        print(f"  end (main) writeVideo()")


    def writeVideoGrouper_v2(self, returnList, i, g, tmpChunks, tmpName, folderName, vidList, filters):
        clamp = min( i + g, len(tmpChunks) - 1 )
        if i < clamp:
            print(f"({tmpName[:9]}) writeVideoGrouper {i} - {clamp}")
            if len(tmpChunks[i:clamp]) > 0:
                _v = list()
                k = i
                tmpMpyVideos = dict()
                #commands = list()
                #filters = dict()
                count = 0
                while k < clamp:
                    if tmpChunks[k][0]:
                        j = k
                        while tmpChunks[k][0] and tmpChunks[k][1] == tmpChunks[j][1] and k < len(tmpChunks) - 1:
                            k += 1
                        video = None
                        #try:
                        if tmpChunks[k][1] not in tmpMpyVideos:
                            video = vidList[tmpChunks[k][1]].getFullVideo()
                            tmpMpyVideos[tmpChunks[k][1]] = video
                        else:
                            video = tmpMpyVideos[tmpChunks[k][1]]
                        a = max(0, tmpChunks[j][4] / 1000)
                        b = min( video.duration, tmpChunks[k][5] / 1000 )
                        #fps = video.fps
                        if vidList[tmpChunks[k][1]].aPath() not in filters.keys():
                            filters[vidList[tmpChunks[k][1]].aPath()] = list()
                            #if COPY_CODEC:
                            #    filters[vidList[tmpChunks[k][1]].aPath()].append( f'-ss {a} -to {b}' )
                            #else:
                            #filters[vidList[tmpChunks[k][1]].aPath()].append( f"between(t,{a},{b})" ) #between a and b
                            #filters[vidList[tmpChunks[k][1]].aPath()].append( f'-ss {a} -to {b}' ) #between a and b
                        #else:
                        ######filters[vidList[tmpChunks[k][1]].aPath()].append( (a, f"+between(t,{a},{b})") ) #between a and b
                        #if COPY_CODEC:
                        #    filters[vidList[tmpChunks[k][1]].aPath()].append( f'-ss {a} -to {b}' )
                        #else:
                        filters[vidList[tmpChunks[k][1]].aPath()].append( (a, (a, b)) ) #between a and b
                            #filters[vidList[tmpChunks[k][1]].aPath()].append( (a, b) ) #between a and b
                            #filters[vidList[tmpChunks[k][1]].aPath()].append( f'-ss {a} -to {b}' ) #between a and b
                        #_v.append( video.subclip( min(a, b), max(a, b) ).set_fps(fps) )
                        #clip = outD.append(f"{folderName}_threading").append(tmpName + f" chunk n{count}.mp4")
                        #commands.append("ffmpeg -i \"{0}\" -c copy -ss {1} -to {2} \"{3}\""
                        #    .format( vidList[tmpChunks[k][1]].aPath(), a, b,
                        #        clip.aPath() ))
                        #returnList.append( (i//g, count, clip, fps) )
                        #returnList.append( (i//g, count, a, b) )
                        count += 1
                    k += 1


    # this function is calculated
    # keep
    def writeVideo_v3(self, tmpChunks, vidList):
        print(f"begin (main) writeVideo()")
        import threading
        threads = list()
        returnList = list()
        nConcurrentThreads=1+(len(tmpChunks)//self.chulenms//10)
        sparsity = len(tmpChunks)//nConcurrentThreads
        folderName = randomString(9)

        count = 0
        filters = dict()
        for i in range( 0, len(tmpChunks), sparsity ):
            if len(threads) > 18: #14
                threads[0].join()
                threads.pop(0) #randomString(9)
            t = threading.Thread(target=self.writeVideoGrouper_v2, args=(returnList, i, sparsity, tmpChunks, "n" + str(count), folderName, vidList, filters, ))
            threads.append(t)
            t.start()
            count += 1

        for thread in threads:
            thread.join()


        count = 0
        fileNames = list()
        codec = '-c:v hevc_nvenc'
        if COPY_CODEC:
            codec = '-c copy'
        if '-cpu' in sys.argv:
            codec = '-c:v h264'
        for a, b in filters.items():
            if COPY_CODEC:
                b = list(map(lambda x: x[1], b))
                step = 1
                sections = list()
                count = 0
                tmpV = mpye.VideoFileClip(a)
                #_fps = mpye.VideoFileClip(a).fps
                for i in range(0, len(b)):
                    f = ffmpeg.input(a)
                    #sections.append( f.trim(start_frame=round(b[i][0]*_fps), end_frame=round(b[i][1]*_fps)) ) #f.subclip(b[i][0], b[i][1]))
                    sections.append( tmpV.subclip( b[i][0], b[i][1] ) )
                    count += 1
                tmpName = outD.append(f"{folderName}_threading").append("{:0>10d}".format(count) + '.mp4').aPath()
                #( ffmpeg.concat(*sections).output(tmpName).run() )
                sg = mpye.concatenate_videoclips(sections, method='compose')
                sg.write_videofile(tmpName)
                fileNames.append( f"file \'{tmpName}\'\n" )

            elif '-speed' in sys.argv: #speed up silent parts instead of trimming
                b = list(map(lambda x: x[1], sorted(b)))
                step = 500
                for i in range(0, len(b), step):
                    end = min(len(b), i + step)
                    tmpName = outD.append(f"{folderName}_threading").append(randomString(24) + '.mp4').aPath()
                    stamps = list()
                    for j in range(i, end, 2): #first, second in b[i:end]:
                        stamps.append(f'+between(t,{j},{j + 1})')
                    stamps = ''.join(stamps)[1:]
                    command = f"ffmpeg -i \"{a}\" -vf \"select=\'{ stamps }\', setpts=N/FRAME_RATE/TB\" -af \"aselect=\'{ stamps }\', asetpts=N/SR/TB\" {codec} \"{tmpName}\"" #-c:v hevc_nvenc
                    p = subprocess.Popen(command)
                    p.communicate()
                    fileNames.append( f"file \'{tmpName}\'\n" )
            elif '-separate_streams' in sys.argv:
                b = list(map(lambda x: x[1], sorted(b)))
                step = 500
                for i in range(0, len(b), step):
                    end = min(len(b), i + step)
                    rString = randomString(24)
                    tmpName = outD.append(f"{folderName}_threading").aPath() + "/" + rString
                    stamps = list()
                    audioSubClips = list()
                    audioSource = mpye.AudioFileClip(a)
                    for first, second in b[i:end]:
                        stamps.append(f'+between(t,{first},{second})')
                        audioSubClips.append(audioSource.subclip(first,second))
                    stamps = ''.join(stamps)[1:]

                    # do audio
                    audioStream = concatenate_audioclips(audioSubClips)
                    audioStream.write_audiofile(tmpName + "_audio.mp3")
                    ######compo = CompositeAudioClip([aclip1.volumex(1.2),aclip2.set_start(5),aclip3.set_start(9)])

                    # do video
                    command = f"ffmpeg -i \"{a}\" -vf \"select=\'{ stamps }\', setpts=N/FRAME_RATE/TB\" {codec} -an \"{tmpName}_video.mp4\""
                    p = subprocess.Popen(command)
                    p.communicate()

                    # join streams
                    command = f"ffmpeg -i \"{tmpName}_video.mp4\" -i \"{tmpName}_audio.mp3\" -c:v copy -c:a aac \"{tmpName}.mp4\""
                    p = subprocess.Popen(command)
                    p.communicate()
                    fileNames.append( f"file \'{tmpName}.mp4\'\n" )
            else:
                b = list(map(lambda x: x[1], sorted(b)))
                step = 100#500
                for i in range(0, len(b), step):
                    end = min(len(b), i + step)
                    tmpName = outD.append(f"{folderName}_threading").append(randomString(24) + '.mp4').aPath()
                    stamps = list()
                    for first, second in b[i:end]:
                        stamps.append(f'+between(t,{first},{second})')
                        #stamps.append(f'+between(t,{round(first,4)},{round(second,4)})')
                    stamps = ''.join(stamps)[1:]
                    #stamps = ''.join(b[i:end])[1:]
                    #stamps = ' '.join(b[i:end])
                    command = f"ffmpeg -i \"{a}\" -vf \"select=\'{ stamps }\', setpts=N/FRAME_RATE/TB\" -af \"aselect=\'{ stamps }\', asetpts=N/SR/TB\" {codec}  -async 1 -crf 7 \"{tmpName}\"" #-c:v hevc_nvenc
                    #command = f"ffmpeg -i \"{a}\" -c copy {stamps} \"{tmpName}.mp4\""
                    p = subprocess.Popen(command)
                    p.communicate()
                    fileNames.append( f"file \'{tmpName}\'\n" )
                    #returnList.append( (a, tmpName) )
                #if True or len(b) > step:


        f = open(outD.append(f"{folderName}_threading").append("list.txt").aPath(),"w+")
        f.write( ''.join(fileNames) )
        #for line in fileNames:
        #    f.write( line )
        f.close()

        command = "ffmpeg -f concat -safe 0 -i \"" + "list.txt" + "\" -c copy \"" + "../" + f'{folderName} -- output.mp4' + "\"" #randomString(5)
        p = subprocess.Popen(command, cwd=outD.append(f"{folderName}_threading").aPath())
        p.communicate()

        if True:
            outD.append(f"{folderName}_threading").delete()
            #for i, j, path in sorted(returnList):
            #    path.delete()

        print(f"  end (main) writeVideo()")



    def runCode(self, sessID, inD, workD, outD, vidList=None):
        print('bp')
        self.sessID = sessID
        self.inD = inD
        self.workD = workD
        if vidList == None:
            vidList = self.vidList(inD)
        spreadCalc = DEFAULT_REACH_ITER
        self.chulenms = DEFAULT_PERIOD
        self.chuLenS = self.chulenms / 1000
        apList = list()
        videoChunks = list()
        tmpVideoChunks = list()
        length = len(vidList)
        import concurrent
        import concurrent.futures
        import threading
        #print("extracting audio")

        # create chunk lists
        #self.chunkList = list()
        #self.tmpCounter = 0
        #print("chunking audio")


        #for i in range(length):
        #    self.extractAudio(i, vidList)
        #    self.chunkAudio(vidList[i])
        # run code once chunks are ready
        print(f"vidList [s={len(vidList)}] = {vidList}")

        UUID = list()

        for i in range( len(vidList) ):
            UUID.append(randomString(7))

        # make audio

        t1 = datetime.datetime.now()
        audioFileList = list()
        for i in range( len(vidList) ):
            self.makeAudio( i, vidList, audioFileList, UUID[i] )
        #print(f"audio file list: {audioFileList}")
        t2 = datetime.datetime.now()
        print(f"self.makeAudio delta: {t2-t1}")



        t1 = datetime.datetime.now()
        audioChunksList = list()

        #executor = concurrent.futures.ProcessPoolExecutor(61)
        #futures = [executor.submit(self.convertMP3toChunksOrganizer, group, #audioFileList, audioChunksList, UUID[i])
        #   for group in grouper( 5, range( len(audioFileList) ) ) ]
        #concurrent.futures.wait(futures)

        for i in range( len(audioFileList) ):
            self.convertMP3toChunks( i, audioFileList, audioChunksList, UUID[i] )
        #print(f"audio chunk list: {audioChunksList}")
        ######exit(1)
        t2 = datetime.datetime.now()
        print(f"self.convertMP3toChunks delta: {t2-t1}")

        #print(f"audio chunks = ")
        #for v in audioChunksList:
        #    print("{}".format(v))
        #exit()


        print(f"flooring values...")
        t1 = datetime.datetime.now()
        audioChunksList = list(map( lambda x: ( x[0], x[1], -500, x[3], x[4] ) if x[2] < -500 else x , audioChunksList ))
        t2 = datetime.datetime.now()
        print(f"flooring audio delta: {t2-t1}")

        #print(f"audio chunks = ")
        #for v in audioChunksList:
        #    print("{}".format(v))
        #exit()


        print(f"calculating spread value: {0}/{len(audioChunksList)}\r", end="")
        audioDataList = [None] * len(audioChunksList)

        t1 = datetime.datetime.now()
        for i in range( len(audioChunksList) ):
            self.helperCalculateSV(i, audioChunksList, audioDataList, -500)
        print(f"\r\n")
        t2 = datetime.datetime.now()
        print(f"self.helperCalculateSV delta: {t2-t1}")


        #print(f"audio data = ")
        #for v in audioDataList:
        #    print("{}".format(v))
        #exit()


        t1 = datetime.datetime.now()

        getVolumes = list( map( lambda x: x[2], audioDataList ) )
        print(f"volumes:\n max = {max(getVolumes)}\n min = {min(getVolumes)}\n avg = {sum(getVolumes)/len(getVolumes)}")

        getSpreadVolumes = list( map( lambda x: x[5], audioDataList ) )
        print(f"spread volumes:\n max = {max(getSpreadVolumes)}\n min = {min(getSpreadVolumes)}\n avg = {sum(getSpreadVolumes)/len(getSpreadVolumes)}")
        #exit() -500 -400
        vt = DEFAULT_THRESHOLD * ( max(getVolumes) + ( sum(getVolumes) / len(getVolumes) ) ) / 2
        vrt = DEFAULT_REACH_THRESH * ( max(getSpreadVolumes) + ( sum(getSpreadVolumes) / len(getSpreadVolumes) ) ) / 2
        filteredChunksList = list( map(lambda x: (True, *x) if x[2] > vt or x[5] > vrt else (False, *x), audioDataList) )
        t2 = datetime.datetime.now()
        print(f"filtering chunks delta: {t2-t1}")

        getTF = list( map( lambda x: x[0], filteredChunksList ) )
        nt = 0
        nf = 0
        for v in getTF:
            if v:
                nt += 1
            else:
                nf += 1
        print(f"\n===\n# true = {nt}\n# false = {nf}\n===\n")

        #print(f"filtered chunks = ")
        #for v in filteredChunksList:
        #    print("{}".format(v))
        #exit()

        ######## https://video.stackexchange.com/questions/10396/how-to-concatenate-clips-from-the-same-video-with-ffmpeg
        idea = """

        ffmpeg -i input -filter_complex \
        "[0:v]trim=60:65,setpts=PTS-STARTPTS[v0]; \
        [0:a]atrim=60:65,asetpts=PTS-STARTPTS[a0]; \
        [0:v]trim=120:125,setpts=PTS-STARTPTS[v1];
        [0:a]atrim=120:125,asetpts=PTS-STARTPTS[a1]; \
        [v0][a0][v1][a1]concat=n=2:v=1:a=1[out]" \
        -map "[out]" output.mkv

        or

        $ ffmpeg -ss 60 -i input -t 5 -codec copy clip1.mkv
        $ ffmpeg -ss 120 -i input -t 5 -codec copy clip2.mkv
        $ echo "file 'clip1.mkv'" > concat.txt
        $ echo "file 'clip2.mkv'" >> concat.txt
        $ ffmpeg -f concat -i concat.txt -codec copy output.mkv

        """

        instructions = f"ffmpeg -i \"{filteredChunksList[i][1]}\" -filter_complex "
        instructions += f"-map \"{randomString(3)} -- output.mp4\" output.mkv"

        referenceVideos = list()
        for i in range(len(vidList)):
            referenceVideos.append(vidList[i].getFullVideo())

        mode = 'multi'

        if mode == 'multi':
            t1 = datetime.datetime.now()
            #def writeVideo_v2(self, tmpChunks, referenceVideos, vidList):
            self.writeVideo_v3(filteredChunksList, vidList)
            t2 = datetime.datetime.now()
            print(f"video write multithread delta: {t2-t1}")
            #outputMovie = mpye.concatenate_videoclips(videoChunksList)


        #oldcoderemove = """
        #write_to_disk = True
        if mode == 'single':

            #
            t1 = datetime.datetime.now()
            videoChunksList = list()
            t1 = datetime.datetime.now()
            i = 0
            while i < len(filteredChunksList):
                if filteredChunksList[i][0]:
                    j = i
                    while filteredChunksList[i][0] and filteredChunksList[i][1] == filteredChunksList[j][1] and i < len(filteredChunksList) - 1:
                        i += 1
                    video = referenceVideos[ filteredChunksList[i][1] ]
                    a = max(0, filteredChunksList[j][4] / 1000)
                    b = min( video.duration, filteredChunksList[i][5] / 1000 )
                    fps = video.fps
                    videoChunksList.append( video.subclip( min(a, b), max(a, b) ).set_fps(fps) ) # fixes mpye bug with frame rounding up
                i += 1
                print(f"creating video clips: {i}/{len(filteredChunksList)}\r", end="")

            print()
            t2 = datetime.datetime.now()
            print(f"subclip gathering delta: {t2-t1}")
            #


            t1 = datetime.datetime.now()
            tagName = randomString(6)
            #executor = concurrent.futures.ProcessPoolExecutor(20)
            #futures = [executor.submit( videoChunksList[i].resize(width=1080).write_videofile(outD.append(f'[{i}]{tagName} -- output.mp4').aPath(), preset='fast') )
            #   for i in range( 0, len(videoChunksList) // 30 )]
            #concurrent.futures.wait(futures)
            #try:
            outputMovie.write_videofile(outD.append(f'{randomString(5)} -- output.mp4').aPath(), preset='medium', audio_codec='aac', threads=16) #, logger=None) #codec='libx265', audio_codec='libmp3lame', audio_bitrate='48k', preset='fast') #, threads=16)
            #except:
            #    print(f"failed to write final video... might need to set a lower threshold")
            t2 = datetime.datetime.now()
            print(f"video write singlethread delta: {t2-t1}")

        #else:
            #outputMovie.resize(width=720).preview(fps=25)
        #"""

        # clean up files including MPY_wvf_snd


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

if __name__ == "__main__":
    #init
    FFMPEG_BIN = 'ffmpeg'
    dir = ''
    dir = dirname(abspath(__file__)) + "\\footage"
    chunks = []
    gen = []

    file = open('options_kss5.json', 'r')
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
    individual_videos = "-foreach" in sys.argv
    COPY_CODEC = '-copy' in sys.argv
    SILENCE_NOTIFICATIONS = '-silent' in sys.argv

    #console_settings
    verbose = data['console_settings']["verbose"]
    print_color = "cyan" #['console_settings']['default_print_color']
    print_color_error = "yellow" #['console_settings']['default_error_color']

    #print('\n=========\n\nsSafe Exit\n\n=========\n')
    #exit()
    sessID = randomString(4)



    workD = kPath(dirname(abspath(__file__)) + "/footage")
    inD = data['file_management_options']['default_input_folder']
    if '[workD]' in inD:
        inD = inD.replace('[workD]\\', '').replace('[workD]', '')
        inD = workD.append(inD)
    else:
        inD = kPath(inD)
    outD = workD.append('output')

    print(bordered(f'{program_name} version {program_version}'))
    import datetime
    a = datetime.datetime.now()
    if individual_videos:
        fSet = list(os.listdir(inD.aPath()))
        print(fSet)
        vids = sorted(list(filter(lambda v: v[-4:].lower() in extList, fSet)))
        vids = list(map(lambda v: inD.append(v), vids))
        print(f'bp ({len(vids)})')
        for v in vids:
            x = kss()
            print('next step')
            x.runCode(sessID, inD, workD, outD, [v])
