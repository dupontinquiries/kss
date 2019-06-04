from __future__ import unicode_literals
import ffmpeg
import os
from os.path import dirname, abspath
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import *
from math import *

import subprocess as sp
import numpy
import sys

# include standard modules
import argparse

#name and version
name = "The Kitchen's Media Tools"
version = "alpha v0.1"

# initiate the parser
parser = argparse.ArgumentParser()
parser.add_argument("-V", "--version", help="show program version", action="store_true")
parser.add_argument("-out", "--output", help="re-render a file", action="store_true")
parser.add_argument("-outf", "--outputfolder", help="re-render all files in a folder", action="store_true")
parser.add_argument("-V", "--version", help="show program version", action="store_true")
parser.add_argument("-V", "--version", help="show program version", action="store_true")
parser.add_argument("-V", "--version", help="show program version", action="store_true")

# read arguments from the command line
args = parser.parse_args()

# check for --version or -V
if args.version:
    print("this is ", name, " version", version)
def render_m(indir, outdir, oxt):
    print()

# take a video and audio and render them out
# example = os.chdir() ... render_s(input['v'], input['a'], 'processed_clip' /or inherit/, '.mp4' /or inheirit/, ['.mp4', .mov] /filter only these filetypes/)
def render_s(v, a, output, outdir, outpath, oxt):
    print()
if args.out:
    render_s()
# take a fodler of videos and render them out
# example = os.chdir() ... render_m('dir_a', 'dir_b', 'processed_', '.mp4' /or inheirit/, ['.mp4', .mov] /filter only these filetypes/)
def render_m(indir, outdir, oprefix, oxt, filter_ext):
    print()
if args.outf:
    render_m()

command = sys.argv[0]
# extension, directory,


FFMPEG_BIN = 'ffmpeg'
dir = dirname(dirname(abspath(__file__))) or "C:\\Users\\kessl\\Desktop\\Code 2019\\kss\\kss\\footage"
chunked_clips = []


#functions for obtaining file info
def duration_of():
    print()
