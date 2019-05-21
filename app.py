from __future__ import unicode_literals
import ffmpeg
import os
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import *
from math import *


#functions

# subtitles and dynamic cuts
def convo_splice(i):
    print("func")

# double filter process
def process_in_groups(i):
    i = str(i.replace(".mp4", ""))
    #
    input = ffmpeg.input(i + ".mp4")
    movie = VideoFileClip(i + ".mp4")
    a = input['a']
    a = a.filter('highpass', 400).filter("lowpass", 15000).filter("loudnorm")
    v = input['v']
    output = ffmpeg.output(a, "tmp_a_from_" + i + ".wav")
    ffmpeg.run(output)
    a = AudioSegment.from_wav("tmp_a_from_" + i + ".wav")
    if os.path.exists("tmp_a_from_" + i + ".wav"):
        os.remove("tmp_a_from_" + i + ".wav")
    chunk_length_ms = 700  # 4000 is best
    chunk_length_s = chunk_length_ms/1000
    chunks_a = make_chunks(a, chunk_length_ms)
    tc_v = []
    list_of_db = []
    list_of_db_solo = []
    spread = 5
    spread_calc = int(((spread - 1) / 2))
    #get start
    for z in range(0, spread_calc):
        list_of_db_solo.append(chunks_a[z])
    list_of_db.append(sum(chunks_a[0:spread]))
    #middle
    for z in range(spread_calc, len(chunks_a) - 1 - spread_calc):
        db = 0
        db_arr = 0
        for q in range(z - spread_calc, z + 1 + spread_calc):
            db_arr += chunks_a[q].dBFS
        db = db_arr/spread
        db_solo = chunks_a[z].dBFS
        list_of_db.append(db)
        list_of_db_solo.append(db_solo)
        print(str(db_solo))
    #get end
    for z in range(len(chunks_a) - 1 - spread_calc, len(chunks_a) - 1):
        list_of_db_solo.append(chunks_a[z])
    list_of_db.append(sum(chunks_a[len(chunks_a) - 1 - spread:len(chunks_a) - 1]))
    max_db = max(list_of_db_solo)
    thresh = 1.2 * max_db
    print("max_db: " + str(max_db))
    print("thresh: " + str(thresh))
    if len(chunks_a) > 1:
        for x in range(0, len(chunks_a) - 1 - spread_calc):
            # op1 - harsh analysis on long pieces
            raw = list_of_db[x] # group
            raw_solo = list_of_db_solo[x]  # group
            if raw > thresh or raw_solo > thresh * .85: # -27 works best, -36
                print("subclips: " + str(tc_v))
                tc_v.append(movie.subclip((x * chunk_length_s), (x * chunk_length_s + chunk_length_s)))
                print("vol_a = " + str(raw))
            # op2 - lenient analysis of shorter lengths
        # come back and check if deleting first skipped ones is necessary
    #finish
    processed = concatenate(tc_v)
    print("concat: " + str(processed))
    processed.write_videofile("final\\processed_output_from_" + i + ".mp4")
    if not processed:
        processed = ffmpeg.output([], [], "final\\processed_output_from_" + i + ".mp4")
        ffmpeg.run(processed)
    ret = ffmpeg.input("final\\processed_output_from_" + i + ".mp4")
    #if os.path.exists("processed_output_from_" + i + ".mp4"):
        #os.remove("processed_output_from_" + i + ".mp4")
    return ret


# make new function that takes the cut times and adds timewarping

# make new function that takes a song and uses the song to determine threshold at the time
# and the cut speed is determined by the time between closest to min and closest to max point (distance) in array

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
    a = a.filter('highpass', 400).filter("lowpass", 15000).filter("loudnorm")
    v = input['v']
    output = ffmpeg.output(a, "tmp_a_from_" + i + ".wav")
    ffmpeg.run(output)
    a = AudioSegment.from_wav("tmp_a_from_" + i + ".wav")
    if os.path.exists("tmp_a_from_" + i + ".wav"):
        os.remove("tmp_a_from_" + i + ".wav")
    chunk_length_ms = 2500  # 4000 is best
    chunk_length_s = chunk_length_ms/1000
    chunks_a = make_chunks(a, chunk_length_ms)
    tc_v = []

    list_of_db = []
    for z in range(0, len(chunks_a) - 1):
        db = chunks_a[z].dBFS
        list_of_db.append(db)

    max_db = max(list_of_db)
    thresh = 1.18 * max_db
    print("max_db: " + str(max_db))
    print("thresh: " + str(thresh))
    if len(chunks_a) > 1:
        for x in range(0, len(chunks_a) - 1):
            raw = list_of_db[x]
            if raw > thresh: # -27 works best, -36
                print("subclips: " + str(tc_v))
                tc_v.append(movie.subclip((x * chunk_length_s), (x * chunk_length_s + chunk_length_s)))
                print("vol = " + str(raw))
                # base = ffmpeg.concat(base_v, base_a, chunks_a[x], tc_v, v=1, a=1)
                # print(str(tc_v))
    # output = ffmpeg.output(base['v'], base['a'], "processed_output_from_" + i + ".mp4")
    # ffmpeg.run(output)
    processed = concatenate(tc_v)
    print("concat: " + str(processed))
    processed.write_videofile("final\\processed_output_from_" + i + ".mp4")
    if not processed:
        processed = ffmpeg.output([], [], "final\\processed_output_from_" + i + ".mp4")
        ffmpeg.run(processed)
    ret = ffmpeg.input("final\\processed_output_from_" + i + ".mp4")
    #if os.path.exists("processed_output_from_" + i + ".mp4"):
        #os.remove("processed_output_from_" + i + ".mp4")
    return ret


print(str(ffmpeg) + " is running")

dir = "C:\\Users\\bluuc\\Desktop\\Code 2019\\Eclipse\\KitchenSplitSilence\\footage"
print("root: " + str(dir))
os.chdir(dir)
vid_arr = create_video_list(dir)
vid_arr.sort(key=lambda x: os.path.getmtime(x))
print("list: " + str(vid_arr) + "")
#base = ffmpeg.input(vid_arr[0])
base = process_in_groups(vid_arr[0])
base_v = base['v']
base_a = base['a']
for w in range(1, len(vid_arr) - 1):
    # concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
    to_concat = process_in_groups(vid_arr[w])
    to_concat_v = to_concat['v']
    to_concat_a = to_concat['a']
    print("n = " + str(w))
    base = ffmpeg.concat(base_v, base_a, to_concat_v, to_concat_a, v=1, a=1)
    base = base.node
    base_v = base['v']
    base_a = base['a']
base_a = base_a.filter("loudnorm").filter("acompressor") # .filter("dynaudnorm")
base_v = base_v.filter("atadenoise").filter("saturation").filter("brightness")
# https://ffmpeg.org/ffmpeg-filters.html#toc-acompressor
#
# https://ffmpeg.org/ffmpeg-filters.html#silencedetect
# base_a = ffmpeg.input("PowerStone.mp3")['a']
out = ffmpeg.output(base_v, base_a, "final\\output_from_" + "all_clips" + ".mp4")
ffmpeg.run(out)
print("done")
# to fix len issue? https://stackoverflow.com/questions/19182188/how-to-find-the-length-of-a-filter-object-in-python
# sort video files by date filmed
# https://stackoverflow.com/questions/168409/how-do-you-get-a-directory-listing-sorted-by-creation-date-in-python
