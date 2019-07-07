from __future__ import unicode_literals
import statistics
import cv2
import ffmpeg
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
import os
from os.path import dirname, abspath
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import *
import moviepy as moviepy
from math import *
import numpy
from termcolor import colored
import tensorflow as tf
import torch as torch

from numba import vectorize

dir = dirname(abspath(__file__)) + "\\footage"
print("root: " + str(dir))
os.chdir(dir)
processed = concatenate([VideoFileClip("a" + ".mp4"), VideoFileClip("b" + ".mp4"), VideoFileClip("c" + ".mp4")])
#export clip
processed.write_videofile("final\\processed_output_from_" + "concat" + ".mp4")
movie = VideoFileClip("final\\processed_output_from_" + "concat" + ".mp4")

if file_size(filename) >= (10 ** 9):
    print(
        "file " + str(filename) + " is large (" + str(file_size(filename)) + ").  Keeping the chunked clips as \"cc\"")
    input = ffmpeg.input(filename)
    name_split = filename.split(".")[0]  # now can only have one period in whole name
    output = ffmpeg.output(input, "compressed_from_large_file_" + name_split + ".mp4")
    render_(output)
    os.rename(filename, 'large_files\\' + filename)
    print("moved " + str(filename) + " to the large_files folder")
    filename = "compressed_from_large_file_" + name_split + ".mp4"