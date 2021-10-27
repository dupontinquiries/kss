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

inVid = "K:\\360\\soundscapes_v1.mp4"
outFolder = "C:\\Users\\kessl\\Desktop\\Insta360Dump\\cape\\mp4\\resolve\\f"

step = 1
start = 30
#start = 0
maxNum = 35
#maxNum = 1513
subClips = list()
print("run")
for i in range(start, maxNum, step):
    end = min(maxNum, i + step)
    suffix = f"chunk_{i}.mp4"
    tmpName = outFolder + "/" + suffix
    subClips.append(suffix)
    command = f"ffmpeg -y -i \"{inVid}\" -c copy -ss {i} -to {end} \"{tmpName}\""
    p = subprocess.Popen(command)
    p.communicate()
    #print(tmpName)
print("clips")
upFolder = "C:\\Users\\kessl\\Desktop\\Insta360Dump\\cape\\mp4\\resolve\\f"
for clip in subClips:
    print("\"" + clip.replace("\\", "\\\\") + "\"")
