from __future__ import unicode_literals
import ffmpeg
import os
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
from moviepy.editor import *
from math import *
from os.path import dirname, abspath
import statistics
import exifread
import cv2

t_stamps = {}

#functions

# subtitles and dynamic cuts
def convo_splice(i):
    print("func")

def get_frame_rate(filename):
    if not os.path.exists(filename):
        sys.stderr.write("ERROR: filename %r was not found!" % (filename,))
        return -1
    out = subprocess.check_output(["ffprobe",filename,"-v","0","-select_streams","v","-print_format","flat","-show_entries","stream=r_frame_rate"])
    rate = out.split('=')[1].strip()[1:-1].split('/')
    if len(rate)==1:
        return float(rate[0])
    if len(rate)==2:
        return float(rate[0])/float(rate[1])
    return -1
def k_remove(a):
    if os.path.exists(a):
        os.remove(a)
# double filter process
def process_in_groups(i, mod, c_l, spread, thresh_mod = 0.9, crop_w = 1080, crop_h = 1350):
    #if os.path.exists("final\\processed_output_from_" + i + ".mp4"):
    #    os.remove("final\\processed_output_from_" + i + ".mp4")
    #if os.path.exists("final\\filtered_and_processed_output_from_" + i + ".mp4"):
    #    os.remove("final\\filtered_and_processed_output_from_" + i + ".mp4")
    i = str(i.replace(".mp4", ""))
    if not os.path.exists("tmp_a_from_" + i + ".mp3"):
        input = ffmpeg.input(i + ".mp4")
        a = input['a']
        a_voice = a.filter('highpass', 300).filter("lowpass", 10000).filter("loudnorm")
        a = a.filter('highpass', 400).filter("lowpass", 15000).filter("loudnorm")
        v = input['v']
        movie = VideoFileClip(i + ".mp4")
        output = ffmpeg.output(a, "tmp_a_from_" + i + ".mp3")
        ffmpeg.run(output)
    if not os.path.exists("tmp_voice_opt_from_" + i + ".mp3"):
        output = ffmpeg.output(a, "tmp_voice_opt_from_" + i + ".mp3")
        ffmpeg.run(output)
    #import the new audio
    a = AudioSegment.from_mp3("tmp_a_from_" + i + ".mp3")
    k_remove("tmp_a_from_" + i + ".mp3")
    a_voices = AudioSegment.from_mp3("tmp_voice_opt_from_" + i + ".mp3")
    k_remove("tmp_voice_opt_from_" + i + ".mp3")
    chunk_length_ms = c_l  # 4000 is best
    chunk_length_s = chunk_length_ms/1000
    chunks_a = make_chunks(a, chunk_length_ms)
    chunks_a_voice = make_chunks(a_voices, chunk_length_ms)
    tc_v = []
    list_of_db = []
    list_of_db_solo = []
    spread_calc = int(((spread - 1) / 2))
    # get start
    db_arr = 0
    for q_1 in range(0, 2 * spread_calc):
        db_arr += chunks_a_voice[q_1].dBFS
    db = db_arr / spread
    for z_init in range(0, spread_calc): # removed -1
        list_of_db.append(db)
        list_of_db_solo.append(chunks_a_voice[z_init].dBFS)
        print(str(z_init) + " run " + "start")
    # middle
    for z_mid in range(spread_calc, len(chunks_a) - 1 - spread_calc):
        db = 0
        db_arr = 0
        for q_2 in range(z_mid - spread_calc, z_mid + 1 + spread_calc):
            db_arr += chunks_a_voice[q_2].dBFS
        db = db_arr / spread
        db_solo = chunks_a_voice[z_mid].dBFS
        list_of_db.append(db)
        list_of_db_solo.append(db_solo)
        print(str(z_mid) + " run " + "middle")
    # get end
    db_arr = 0
    for q_3 in range(len(chunks_a) - 1 - spread_calc, len(chunks_a) - 1):
        db_arr += chunks_a_voice[q_3].dBFS
    db = db_arr / spread
    for z_end in range(len(chunks_a) - (2 * spread_calc), len(chunks_a) - 1):
        list_of_db.append(db)
        list_of_db_solo.append(chunks_a_voice[z_end].dBFS)
        print(str(z_end) + " run " + "end")
    max_db = max(list_of_db_solo)
    median_db = statistics.median(list_of_db)
    average_db = statistics.average(list_of_db)
    median_infl_avg_db = (.5 * median_db) + (.5 * average_db)
    thresh = mod * median_infl_avg_db
    print("max_db: " + str(max_db))
    print("thresh: " + str(thresh))
    fps = movie.fps
    print("fps: " + str(len(list_of_db)))
    if len(chunks_a) > 1:
        for x in range(0, len(chunks_a) - 1):
            # op1 - harsh analysis on long pieces
            raw = list_of_db[x]  # group
            raw_solo = list_of_db_solo[x]  # group
            if raw_solo > thresh or raw > (thresh * thresh_mod):
                # print("subclips: " + str(tc_v))
                tmp = movie.subclip((x * chunk_length_s), ((x + 1) * chunk_length_s))
                tc_v.append(tmp)  # this is wrong
                print("vol_a = " + str(raw))
            # op2 - lenient analysis of shorter lengths
        # come back and check if deleting first skipped ones is necessary
    # finish
    processed = concatenate(tc_v)
    print("concat: " + str(processed))
    processed.write_videofile("final\\processed_output_from_" + i + ".mp4")
    ret = ffmpeg.input("final\\processed_output_from_" + i + ".mp4")
    movie_width = movie.w
    movie_height = movie.h
    desired_height = crop_h
    movie_width = movie.w
    desired_width = crop_w
    scale_factor = max((movie_height / desired_height), (movie_width / desired_width))
    print("scale factor")
    print("scale factor = " + scale_factor)
    print("scale factor")
    base_a = ret['a'].filter("loudnorm").filter("acompressor")  # .filter("dynaudnorm")
    ffmpeg.run(ffmpeg.output(base_a, "final\\processed_audio_from" + i + ".mp3"))
    new_audio = (ffmpeg.input("final\\processed_audio_from" + i + ".mp3"))
    base_v = ret['v'].filter("atadenoise").filter('crop', x=0*crop_w*scale_factor, y=0, w=crop_w*scale_factor, h=crop_h*scale_factor)
    output = ffmpeg.output(base_v, new_audio, "final\\filtered_and_processed_output_from_" + i + ".mp4")
    ffmpeg.run(output)
    ret = ffmpeg.input("final\\filtered_and_processed_output_from_" + i + ".mp4")
    #if os.path.exists("final\\processed_output_from_" + i + ".mp4"): os.remove("final\\processed_output_from_" + i + ".mp4") if os.path.exists("final\\filtered_and_processed_output_from_" + i + ".mp4"): os.remove("final\\filtered_and_processed_output_from_" + i + ".mp4")
    return ret



# make new function that takes the cut times and adds timewarping

# make new function that takes a song and uses the song to determine threshold at the time
# and the cut speed is determined by the time between closest to min and closest to max point (distance) in array
def to_mp4(name):
    i = ffmpeg.input(name)
    o = ffmpeg.output(i, 'file--' + name + '--.mp4')
    ffmpeg.run(o)
def create_timestamps(name):
    print('fetching timestamps for ' + name)
    cap = cv2.VideoCapture(name)
    fps = cap.get(cv2.CAP_PROP_FPS)

    timestamps_tmp = [cap.get(cv2.CAP_PROP_POS_MSEC)]

    while (cap.isOpened()):
        frame_exists, curr_frame = cap.read()
        if frame_exists:
            timestamps_tmp.append(cap.get(cv2.CAP_PROP_POS_MSEC))
        else:
            break

    cap.release()

    for i, (ts) in enumerate(zip(timestamps_tmp)):
        print('Frame timestamp from ' + name + ': ' + str(ts))
    t_stamps[name] = timestamps_tmp
def create_video_list(a, ts = True):
    tmp = []
    for name in os.listdir(a):
        if name.endswith(".m2ts") or name.endswith(".mov"):
            to_mp4(name)
            os.rename(name, 'notmp4\\' + name)
    for name in os.listdir(a):
        if name.endswith(".mp4"):
            tmp.append(name)
            #get timestamps
            if ts:
                create_timestamps(name)
    return tmp

def get_length(filename):
  result = subprocess.Popen(["ffprobe", filename], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines() if "Duration" in x]

def process(i):
    #
    input = ffmpeg.input(i)
    movie = VideoFileClip(i)
    og_name = i
    i = str(i.replace(".mp4", ""))
    a = input['a']
    a_voice = a.filter('highpass', 300).filter("lowpass", 10000).filter("loudnorm")
    a = a.filter('highpass', 400).filter("lowpass", 15000).filter("loudnorm")
    v = input['v']
    output = ffmpeg.output(a, "tmp_a_from_" + i + ".wav")
    ffmpeg.run(output)
    output = ffmpeg.output(a, "tmp_voice_opt_from_" + i + ".wav")
    ffmpeg.run(output)
    a = AudioSegment.from_wav("tmp_a_from_" + i + ".wav")
    if os.path.exists("tmp_a_from_" + i + ".wav"):
        os.remove("tmp_a_from_" + i + ".wav")
    a_voices = AudioSegment.from_wav("tmp_voice_opt_from_" + i + ".wav")
    if os.path.exists("tmp_voice_opt_from_" + i + ".wav"):
        os.remove("tmp_voice_opt_from_" + i + ".wav")
    chunk_length_ms = 2500  # 4000 is best
    chunk_length_s = chunk_length_ms/1000
    chunks_a = make_chunks(a, chunk_length_ms)
    chunks_a_voice = make_chunks(a, chunk_length_ms)
    tc_v = []

    list_of_db = []
    for z in range(0, len(chunks_a) - 1):
        db = chunks_a_voice[z].dBFS
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
    ret = ffmpeg.input("final\\processed_output_from_" + i + ".mp4")
    movie_width = movie.width
    movie_height = movie.height
    desired_height = 1350
    scale_factor = movie_height/desired_height
    base_a = ret['a'].filter("loudnorm").filter("acompressor")  # .filter("dynaudnorm")
    base_v = ret['v'].filter("atadenoise").filter('scale', w=(str(movie_width*scale_factor)),
                                                  h=(str(movie_height*scale_factor))).filter('crop', w='1080', h='1350')
    ffmpeg.run(base_v, base_a, "final\\filtered_and_processed_output_from_" + i + ".mp4")
    ret = ffmpeg.input("final\\filtered_and_processed_output_from_" + i + ".mp4")
    #if os.path.exists("processed_output_from_" + i + ".mp4"):
        #os.remove("processed_output_from_" + i + ".mp4")
    return ret


print(str(ffmpeg) + " is running")

dir = dirname(abspath(__file__)) + "\\footage"
print("root: " + str(dir))
os.chdir(dir)
vid_arr = create_video_list(dir, False)
#vid_arr.sort(key=lambda x: os.path.getmtime(x))
print("list: " + str(vid_arr) + "")
#base = ffmpeg.input(vid_arr[0])
base = process_in_groups(vid_arr[0], 0.9, 400, 6, 1.35, 1080, 1350)
base_v = base['v']
base_a = base['a']
if(len(vid_arr) > 1):
    for w in range(1, len(vid_arr) - 1):
        # concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
        to_concat = process_in_groups(vid_arr[w], 0.9, 400, 6, 1.35, 1080, 1350) #1.4, 15000, 5 is pretty good with about 50% retention and near full comprehensibility
        to_concat_v = to_concat['v']
        to_concat_a = to_concat['a']
        print("n = " + str(w))
        base = ffmpeg.concat(base_v, base_a, to_concat_v, to_concat_a, v=1, a=1)
        base = base.node
        base_v = base['v']
        base_a = base['a']
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