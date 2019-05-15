from __future__ import unicode_literals
import ffmpeg
import os
import subprocess
import sys
import logging
import re
from pydub import AudioSegment
from pydub.utils import make_chunks

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
    i = str(i.replace(".mp4", ""))
    a = input['a']
    v = input['v']
    output = ffmpeg.output(a, "tmp_a_from_" + i + ".wav")
    ffmpeg.run(output)
    a = AudioSegment.from_wav("tmp_a_from_" + i + ".wav")
    if os.path.exists("tmp_a_from_" + i + ".wav"):
        os.remove("tmp_a_from_" + i + ".wav")
    chunk_length_ms = 500  # 5000 is best, but one and 250 are also good
    chunks_a = make_chunks(a, chunk_length_ms)
    chunks_v = make_chunks(v, chunk_length_ms)
    for x in range(0, len(chunks_a) - 1):
        raw = chunks_a[x].dBFS
        if raw < -20:  #-12 is best
            print("vol = " + str(raw))
            del chunks_a[x]
            del chunks_v[x]
    output = ffmpeg.output(chunks_v, chunks_a, "processed_output_from_" + i + ".mp4")
    ffmpeg.run(output)
    return ffmpeg.input("processed_output_from_" + i + ".mp4")


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
for w in range(len(vid_arr) - 2):
    #concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
    to_concat = process(vid_arr[w+1])
    to_concat_v = to_concat['v']
    to_concat_a = to_concat['a']
    print("n = " + str(w))
    base = ffmpeg.concat(base_v, base_a, to_concat_v, to_concat_a, v=1, a=1)
    base = base.node
    base_v = base['v']
    base_a = base['a']
base_a = base_a.filter('volume', 0)
out = ffmpeg.output(base_v, base_a, "output_from_" + "all_clips" + ".mp4")
ffmpeg.run(out)
print("done")
# to fix len issue? https://stackoverflow.com/questions/19182188/how-to-find-the-length-of-a-filter-object-in-python
