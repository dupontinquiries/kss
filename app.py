from __future__ import unicode_literals
import ffmpeg
import os
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import *
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.tools.cuts import find_video_period
from moviepy.audio.tools.cuts import find_audio_period
import sys
import logging
import re


#functions

def create_video_list(a):
    tmp = []
    for name in os.listdir(a):
        if name.endswith(".mp4"):
            tmp.append(name)
    return tmp

def get_length(filename):
  result = subprocess.Popen(["ffprobe", filename], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines() if "Duration" in x]

#start of code
def process(i):
    #
    input = ffmpeg.input(i)
    movie = VideoFileClip(i)
    og_name = i
    i = str(i.replace(".mp4", ""))
    a = input['a']
    v = input['v']
    output = ffmpeg.output(a, "tmp_a_from_" + i + ".wav")
    ffmpeg.run(output)
    a = AudioSegment.from_wav("tmp_a_from_" + i + ".wav")
    if os.path.exists("tmp_a_from_" + i + ".wav"):
        os.remove("tmp_a_from_" + i + ".wav")
    chunk_length_ms = 500  # 5000 is best, but one and 250 are also good
    chunk_length_s = chunk_length_ms/1000  # 5000 is best, but one and 250 are also good
    chunks_a = make_chunks(a, chunk_length_ms)
    tc_v = []
    for x in range(0, len(chunks_a) - 1):
        raw = chunks_a[x].dBFS
        if raw > -30:  # -12 is best
            print("subclips: " + str(tc_v))
            tc_v.append(movie.subclip((x * chunk_length_s), (x * chunk_length_s + chunk_length_s)))
            print("vol = " + str(raw))
            # base = ffmpeg.concat(base_v, base_a, chunks_a[x], tc_v, v=1, a=1)
            # print(str(tc_v))
    # output = ffmpeg.output(base['v'], base['a'], "processed_output_from_" + i + ".mp4")
    # ffmpeg.run(output)
    processed = concatenate(tc_v)
    processed.write_videofile("processed_output_from_" + i + ".mp4")
    if not processed:
        processed = ffmpeg.output([], [], "processed_output_from_" + i + ".mp4")
        ffmpeg.run(processed)
    ret = ffmpeg.input("processed_output_from_" + i + ".mp4")
    if os.path.exists("processed_output_from_" + i + ".mp4"):
        os.remove("processed_output_from_" + i + ".mp4")
    return ret


print(str(ffmpeg) + " is running")

dir = "C:\\Users\\bluuc\\Desktop\\Code 2019\\Eclipse\\KitchenSplitSilence\\footage"
print("root: " + str(dir))
os.chdir(dir)
vid_arr = create_video_list(dir)
print("list: " + str(vid_arr) + "")
#base = ffmpeg.input(vid_arr[0])
base = process(vid_arr[0])
base_v = base['v']
base_a = base['a']
for w in range(1, len(vid_arr) - 1):
    # concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
    to_concat = process(vid_arr[w])
    to_concat_v = to_concat['v']
    to_concat_a = to_concat['a']
    print("n = " + str(w))
    base = ffmpeg.concat(base_v, base_a, to_concat_v, to_concat_a, v=1, a=1)
    base = base.node
    base_v = base['v']
    base_a = base['a']
base_a = base_a.filter('highpass', 50).filter("lowpass", 18000).filter("loudnorm").filter("acompressor")
# https://ffmpeg.org/ffmpeg-filters.html#toc-acompressor
#
# https://ffmpeg.org/ffmpeg-filters.html#silencedetect
out = ffmpeg.output(base_v, base_a, "output_from_" + "all_clips" + ".mp4")
ffmpeg.run(out)
print("done")
# to fix len issue? https://stackoverflow.com/questions/19182188/how-to-find-the-length-of-a-filter-object-in-python
