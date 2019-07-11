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

t_stamps = {}

#functions

FFMPEG_BIN = 'ffmpeg'
dir = dirname(abspath(__file__)) + "\\footage" or "C:\\Users\\kessl\\Desktop\\Code 2019\\kss\\kss\\footage"
chunked_clips = []
chunked_timestamps = []
clips_to_remove = []

#def vals
DEFAULT_THRESHOLD = 0.98
DEFAULT_PERIOD = 850
DEFAULT_REACH_ITER = 3
DEFAULT_REACH_THRESH = 1.02
DEFAULT_WIDTH = 2560
DEFAULT_HEIGHT = 1440
DEFAULT_MAX_CHUNK_SIZE = 1.20

#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#with tf.device("/gpu:0"):
#print("current device = " + str(torch.cuda.current_device()))
#with torch.cuda.device(1):
def main():
    #w = torch.FloatTensor(2,3).cuda()
    # w was placed in  device_1 by default.
    dir = dirname(abspath(__file__)) + "\\footage"
    print("root: " + str(dir))
    os.chdir(dir)
    vid_arr = create_video_list(dir, False)
    if len(vid_arr) < 1:
        print("no files in directory: " + str(dir))
        #sys.exit(0)
    initial = 0
    process = distr(vid_arr[initial], DEFAULT_THRESHOLD, DEFAULT_PERIOD, DEFAULT_REACH_ITER, DEFAULT_REACH_THRESH, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_MAX_CHUNK_SIZE)
    while process == False and initial < len(vid_arr) - 1:
        initial = initial + 1
        process = distr(vid_arr[initial], DEFAULT_PERIOD, DEFAULT_PERIOD, DEFAULT_REACH_ITER, DEFAULT_REACH_THRESH,
                        DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_MAX_CHUNK_SIZE)
    main = process
    if initial < len(vid_arr) - 1:
        for w in range(initial, len(vid_arr) - 1):
            # concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
            process = distr(vid_arr[w], DEFAULT_PERIOD, DEFAULT_PERIOD, DEFAULT_REACH_ITER, DEFAULT_REACH_THRESH, DEFAULT_WIDTH, DEFAULT_HEIGHT, DEFAULT_MAX_CHUNK_SIZE)
            if process != False:
                main = k_concat([main, process])
    out = ffmpeg.output(main, "final\\output_from_all_clips.mp4")
    render_(out) #format='h264'
    #clean up all clips
    for clip in clips_to_remove:
        os.remove(str(clip))
    print("done")

def file_size(fname):
    statinfo = os.stat(fname)
    return statinfo.st_size

def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def read_in_ffmpeg_chunks(filename, max_chunk_size, file_length):
    file_length = floor(file_length)
    max_chunk_size *= 60
    t_s = 0
    t_f = max_chunk_size
    while file_length - t_s > 0:
        delta = t_f - t_s
        print("t_s = " + str(t_s) + "; " + "t_f = " + str(t_f) + "; " + "d = " + str(delta) + "; ")
        #if round(delta * 10) / 10 < 1 or round((file_length - t_f) * 10) / 10 < 1:
        if file_length - t_f <= 0:
            yield False
        name = str('moviepy_subclip_' + str(t_s) + '_from_' + str(filename))
        #try
        sub = moviepy.video.io.ffmpeg_tools.ffmpeg_extract_subclip(str(filename), t_s, t_f, targetname=name)
        clips_to_remove.append(name)
        t_s += delta
        t_f += delta
        ret = ffmpeg.input(name)
        if not ret:
            yield False
        yield [ret, name]

def k_remove(a):
    if os.path.exists(a):
        os.remove(a)

def k_concat(a):
    b = a[0]
    if len(a) == 0:
        return None
    if len(a) == 1:
        return b
    else:
        for x in range(1, len(a) - 1):
            c = a[x]
            b = ffmpeg.concat(b, c)
        return b

def distr(filename, mod, c_l, spread, thresh_mod = 0.9, crop_w = 1080, crop_h = 1350, max_chunk_size = 5):
    # compress any large files
    smaller_clips = []
    print("attempting to distr() " + filename)
    if  "completed_file_" in filename: #or "moviepy_subclip" in filename:
        return False
    tmp_clip = VideoFileClip(filename)
    l = tmp_clip.duration
    tmp_clip.close()
    del tmp_clip
    print("file length = " + str(l))
    for piece in read_in_ffmpeg_chunks(filename, max_chunk_size, l):
        if piece is not False:
            if file_size(filename) >= (10 ** 9):
                print("file " + str(filename) + " is large (" + str(
                    file_size(filename)) + ").  (Future Capability) Keeping the chunked clips as \"cc\"")
            print('piece: ' + str(piece))
            result = process_audio_loudness_over_time(piece[0], piece[1], mod, c_l, spread, thresh_mod, crop_w, crop_h)
            if result is not False:
                smaller_clips.append(result)
    if len(smaller_clips) >= 1:
        #total = ffmpeg.merge_outputs(smaller_clips)
        #total = ffmpeg.concat(smaller_clips)
        total = k_concat(smaller_clips)
        output = ffmpeg.output(total, 'completed_file_of_' + filename)
        render_(output)
        return ffmpeg.input('completed_file_of_' + filename)
    else:
        return False

def render_(component):
    ffmpeg.run(component, overwrite_output=True)

def floor_out(a, bottom):
    if a < bottom:
        return bottom
    else:
        return a

#merge_outputs = combine clips; overwrite_output = overwrite files /save lines of code
def process_audio_loudness_over_time(i, name, mod_solo, c_l, spread, mod_multi, crop_w, crop_h):
    #clean up files space
    #remove .mp4 to use other filetypes like .wav
    name = str(name.replace(".mp4", ""))
    input = i
    #v = input['v']
    a = input['a']
    #clean up audio so program takes loudness of voice into account moreso than other sounds
    #clean up audio of final video
    a = a.filter('highpass', 45).filter("lowpass", 17000).filter("loudnorm").filter("afftdn", nr=6, nt="w", om="o")
    a_voice = a.filter('highpass', 600).filter("lowpass", 7000).filter("loudnorm").filter("afftdn", nr=11, nt="w", om="o")
    clips_to_remove.append("tmp_a_from_" + name + ".wav")
    clips_to_remove.append("voice\\tmp_voice_opt_from_" + name + ".wav")
    print('sub[0] = ' + str(i))
    print('sub[1] = ' + name)
    #get subclips in the processing part
    print("opening " + "final\\processed_output_from_" + name + ".mp4")
    movie_v = VideoFileClip(name + ".mp4")
    duration = movie_v.duration
    #fps = movie_v.fps
    #export clip audio
    output = ffmpeg.output(a, "tmp_a_from_" + name + ".wav")
    render_(output)
    movie_a = AudioFileClip("tmp_a_from_" + name + ".wav")
    duration_a = movie_a.duration
    #fps_a = movie_a.fps
    #export voice_optimized audio
    output = ffmpeg.output(a_voice, "voice\\tmp_voice_opt_from_" + name + ".wav")
    render_(output)
    #import the new audio
    a = AudioSegment.from_mp3("tmp_a_from_" + name + ".wav")
    a_voices = AudioSegment.from_mp3("voice\\tmp_voice_opt_from_" + name + ".wav")
    # remove file that was rendered
    clips_to_remove.append("voice\\tmp_voice_opt_from_" + name + ".wav")
    #create averaged audio sections and decide which ones meet threshold
    chunk_length_ms = c_l
    chunk_length_s = chunk_length_ms/1000
    chunks_a = make_chunks(a, chunk_length_ms)
    chunks_a_voice = make_chunks(a_voices, chunk_length_ms)
    tc_v = []
    tc_a = []
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
    #reformat the sound levels
    floor = 150
    list_of_db = list(map(lambda x: floor_out(x, - floor), list_of_db))
    list_of_db_solo = list(map(lambda x: floor_out(x, - floor), list_of_db_solo))
    list_of_db = list(map(lambda x: x + floor, list_of_db))
    list_of_db_solo = list(map(lambda x: x + floor, list_of_db_solo))
    #get target threshold to use for modifiers
    max_db = max(list_of_db)
    median_db = statistics.median(list_of_db)
    average_db = statistics.mean(list_of_db)
    max_db_solo = max(list_of_db_solo)
    median_db_solo = statistics.median(list_of_db_solo)
    average_db_solo = statistics.mean(list_of_db_solo)
    target_db = ((.15 * median_db_solo) + (.25 * average_db_solo) + (.6 * max_db_solo))
    thresh = mod_solo * target_db
    target_db = ((.3 * median_db) + (.5 * average_db) + (.2 * max_db))
    thresh_multi = mod_multi * target_db
    #logging purposes
    #print("max_db_solo: " + str(max_db_solo) + "/" + floor)
    #print("max_db_multi: " + str(max_db) + "/" + floor)
    print("thresh_solo = " + str(thresh))
    print("thresh_multi = " + str(thresh_multi))
    print("clip duration = " + str(duration))
    if len(chunks_a) > 1:
        for x in range(0, len(chunks_a) - 1):
            # op1 - harsh analysis on long pieces
            raw = list_of_db[x]  # group
            raw_solo = list_of_db_solo[x]  # group
            if raw_solo > thresh or raw > thresh_multi:
                start = max(0, x * chunk_length_s)
                cap = (x + 1) * chunk_length_s #min((x + 1) * chunk_length_s, duration)
                print("start pre = " + str(start))
                print("cap pre = " + str(cap))
                if cap < duration:
                    tmp_v = movie_v.subclip(start, cap)
                    tc_v.append(tmp_v)
                    tmp_a = movie_a.subclip(start, cap)
                    tc_a.append(tmp_a)
                    tmp_v.close()
                    del tmp_v.reader
                    del tmp_v
                    del tmp_a
                    print(colored("Section #" + str(x)
                                      + "\n" + "peak volume = " + str(raw_solo)
                                      + "\n" + "avg  volume = " + str(raw), 'green'))
                else:
                    print(colored("Failed to render this subclip due to errors with the time parameters!", 'red'))
            else:
                print(colored("Section #" + str(x)
                              + "\n" + "peak volume = " + str(raw_solo)
                              + "\n" + "avg  volume = " + str(raw), 'blue'))
        # combine all clips into one
    if len(tc_v) < 1:
        print(colored("No packets came through.", 'blue'))
        return False
    processed_v = None
    processed_a = None
    if len(tc_v) > 1:
         #invalid handle
        processed_a = moviepy.editor.concatenate_audioclips(tc_a)
        processed_v = moviepy.editor.concatenate_videoclips(tc_v) #error invalid handle
        print(colored("Concatenating accepted packets.", 'green'))
    else:
        processed_v = tc_v[0]
        processed_a = tc_a[0]
        print(colored("Taking the single packet that was accepted", 'green'))
    #export clip
    print("export clip video = " + str(processed_v))
    print("export clip audio = " + str(processed_a))
    processed_v = processed_v.set_audio(processed_a) #issue with audio
    duration = processed_v.duration
    processed_v.write_videofile("final\\processed_output_from_" + name + ".mp4")
    #processed_a.write_audiofile("final\\processed_audio_from_" + name + ".wav")
    clips_to_remove.append("final\\processed_output_from_" + name + ".mp4")
    clips_to_remove.append("final\\processed_audio_from_" + name + ".mp4")
    #reopen combined clip
    print("reopening " + "final\\processed_output_from_" + name + ".mp4")
    #movie_v = VideoFileClip("final\\processed_output_from_" + name + ".mp4")
    #duration = movie_v.duration
    movie_v.reader.close()
    del processed_a
    print("concat duration = " + str(duration))
    #reimport for filtering
    # filter video
    ret = ffmpeg.input("final\\processed_output_from_" + name + ".mp4")
    print("ret = " + str(ret))
    movie_height = movie_v.h
    desired_height = crop_h
    movie_width = movie_v.w
    desired_width = crop_w
    scale_factor = min((movie_height / desired_height), (movie_width / desired_width))
    #print("scale factor = " + str(scale_factor))
    base_v = ret['v'].filter('crop', x=1.50*crop_w*scale_factor, y=0*crop_h*scale_factor, w=crop_w*scale_factor, h=crop_h*scale_factor)
    #filter audio
    #base_a = ffmpeg.input("final\\processed_audio_from_" + name + ".wav")
    base_a = ret['a'].filter("loudnorm").filter("equalizer", f=7000, width_type="o", width=5, g=1).filter("equalizer", f=200, width_type="o", width=2, g=-1)#.filter("equalizer", f=200, width_type="o", width=2, g=1) #.filter("acompressor")
    #output
    output = ffmpeg.output(base_v, base_a, "final\\filtered_and_processed_output_from_" + name + ".mp4")
    render_(output)
    ret = ffmpeg.input("final\\filtered_and_processed_output_from_" + name + ".mp4")
    clips_to_remove.append("final\\processed_output_from_" + name + ".mp4")
    clips_to_remove.append("final\\filtered_and_processed_output_from_" + name + ".mp4")
    return ret

# make new function that takes the cut times and adds timewarping

# make new function that takes a song and uses the song to determine threshold at the time
# and the cut speed is determined by the time between closest to min and closest to max point (distance) in array
def to_mp4(name):
    name_root = name[-4:]
    i = ffmpeg.input(name)
    o = ffmpeg.output(i, 'file_computed_from' + name_root + '.mp4')
    render_(o)
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
        name = name.lower()
        name_root = name[-4:]
        print(name_root)
        if name.endswith(".m2ts") or name.endswith(".mov") or name.endswith(".h264"):
            to_mp4(name)
            os.rename(name, 'not_mp4\\' + name)
    for name in os.listdir(a):
        if name.endswith(".MP4"):
            name = name.replace(".MP4", ".mp4")
        if name.endswith(".mp4"):
            tmp.append(name)
            #get timestamps
            if ts:
                create_timestamps(name)
    return tmp

def get_length(filename):
  result = subprocess.Popen(["ffprobe", filename], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines() if "Duration" in x]

main()