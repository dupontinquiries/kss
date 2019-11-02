from __future__ import unicode_literals
import statistics
import cv2
import ffmpeg
from moviepy.video.io.ffmpeg_tools import *
import moviepy
from moviepy.editor import *
from os.path import dirname, abspath
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
#from moviepy.editor import concatenate_videoclips, concatenate_audioclips, VideoFileClip, AudioFileClip
#from moviepy.editor import *
# from moviepy import write_videofile
# from math import *
# import numpy
from termcolor import colored
# import tensorflow-gpu as tf
import random
import os
import sys
from k_chunk import k_chunk

def render_(component):
    ffmpeg.run(component, overwrite_output=True)

FFMPEG_BIN = 'ffmpeg'
dir = dirname(abspath(__file__)) + "\\footage"
os.chdir(dir)
chunked_clips = []
chunked_timestamps = []
clips_to_remove = []

verbose = True

test_file = 'chunks\\a.wav'

name = 'moviepy_subclip_0_72.0_from_GOPR0174_comp'
name_audio = 'chunks\\tmp_a_from_' + name + '.wav'
input = ffmpeg.input(name_audio)
print(input)

movie_a_fc = AudioSegment.from_wav(test_file)
print(movie_a_fc)

mpy = AudioFileClip(test_file)
mpy.close()
print(mpy)
