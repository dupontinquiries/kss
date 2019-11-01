from __future__ import unicode_literals
import statistics
import cv2
import ffmpeg
from moviepy.video.io.ffmpeg_tools import *
import moviepy
import moviepy.editor as mpye
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
chunked_clips = []
chunked_timestamps = []
clips_to_remove = []

verbose = True

name = 'GOPR0213_comp'
input = ffmpeg.input(name + '.mp4')

name_audio = 'chunks\\tmp_a_from_' + name + '.wav'
name_audio_voice = 'chunks\\tmp_voice_opt_from_' + name + '.wav'
if verbose: print(colored('Preparing audio for video...', 'blue'))
# video clip audio
a_name_audio = input['a']
print(a_name_audio)
# clean up audio so program takes loudness of voice into account moreso than other sounds
# clean up audio of final video
if verbose: print(colored('Preparing tailored audio...', 'blue'))
a_name_audio = a_name_audio.filter("afftdn", nr=12, nt="w", om="o").filter('highpass', 400).filter("lowpass", 3400).filter("loudnorm").filter("afftdn", nr=12, nt="w", om="o")

if verbose: print(colored('Writing tailored audio...', 'blue'))
output = ffmpeg.output(a_name_audio, name_audio)
render_(output)
