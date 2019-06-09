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

#functions

FFMPEG_BIN = 'ffmpeg'
dir = dirname(abspath(__file__)) + "\\footage" or "C:\\Users\\kessl\\Desktop\\Code 2019\\kss\\kss\\footage"
chunked_clips = []
chunked_timestamps = []

def read_audio_data(file_path, offset='00:00:00'):
    #define TRUE 0
    command = [ FFMPEG_BIN,
                '-ss',offset,
                '-i', file_path,
                '-f', 's16le',
                '-acodec', 'pcm_s16le',
                '-ar', '44100', # ouput will have 44100 Hz
                '-ac', '2', # stereo (set to '1' for mono)
                '-']
    pipe = sp.Popen(command, stdout=sp.PIPE, bufsize=10**8)
    return pipe

def write_audio_file(file_path,audio_array):
    '''
    Need to do this just for testing
    want to be able to create some bizzarre test files for croma print
    '''
    pipe = sp.Popen([ FFMPEG_BIN,
       '-y', # (optional) means overwrite the output file if it already exists.
       "-acodec", "pcm_s16le", # means raw 16bit input
       '-r', "44100", # the input will have 44100 Hz
       '-ac','2', # the input will have 2 channels (stereo)
        "-f", 's16le', # means 16bit input
       '-i', '-', # means that the input will arrive from the pipe
       '-vn', # means "don't expect any video input"
       '-acodec', "aac","-strict" ,"-2",#"ac3_fixed", # output audio codec
        '-acodec', "adpcm_sw",#"ac3_fixed", # output audio codec
       '-b',"mp3", # output bitrate (=quality). Here, 3000kb/second
                      file_path],
                    stdin=sp.PIPE,stdout=sp.PIPE, stderr=sys.stdin)

    pipe.stdin.write(audio_array)

def get_audio_array(pipe, minutes):
    number_of_audio_frames = 88200 * 30 * minutes
    bytes_in_frame = 4
    bytes_to_read = number_of_audio_frames * bytes_in_frame
    raw_audio = pipe.stdout.read(bytes_to_read)

    raw_audio_array = numpy.fromstring(raw_audio, dtype="int16")
    if len(raw_audio_array) < 1:
        return None,None,'reached end of file'
    audio_array = raw_audio_array.reshape((len(raw_audio_array)/2,2))
    return audio_array, raw_audio, None

def read_creation_date():
    return 0

def formulate_timestamp(chunk_file_path, max_chunk_size, chunk_count_predecessors):
    return 0

def chunk_file(file_name, max_chunk_size = 10, file_suffix = "default", extention= ".mp3"):
    if file_suffix == "default":
        file_suffix = file_name
    pipe = read_audio_data(file_name)
    clip = VideoFileClip(file_name)
    vl = clip.duration
    i_2 = ceil((vl/60)/(max_chunk_size*1))
    for i in range(0, i_2 - 1):
        start = i * max_chunk_size * 60
        end = min(vl / 60, i * max_chunk_size + 1) * 60
        chunk_file_path = 'k_chunk_n=' + str(i) + '_from_' + file_suffix + extention
        chunked_clips.append(chunk_file_path)
        chunked_timestamps.append([file_name, start, end])
        clip.subclip(start, end).write_videofile(chunk_file_path)
        print('wrote file ' + chunk_file_path)

def chunk_folder(max_chunk_size = 5, folder_name = ""):
    '''
    so basically go through all files in folder
    turn each file into a bunch of files of length max_chunk_size(given in minutes)
    '''
    for name in os.listdir(dir + folder_name):
        if name.endswith(".mp4") and str("k_chunk_n=") not in name:
            chunk_file(name, max_chunk_size)

# double filter process
def process_in_groups(i, mod, c_l, spread, thresh_mod = 0.9, crop_w = 1080, crop_h = 1350):
    if os.path.exists("final\\processed_output_from_" + i + ".mp4"):
        os.remove("final\\processed_output_from_" + i + ".mp4")
    if os.path.exists("final\\filtered_and_processed_output_from_" + i + ".mp4"):
        os.remove("final\\filtered_and_processed_output_from_" + i + ".mp4")
    i = str(i.replace(".mp4", ""))
    #
    input = ffmpeg.input(i + ".mp4")
    a = input['a']
    a_voice = a.filter('highpass', 300).filter("lowpass", 10000).filter("loudnorm")
    a = a.filter('highpass', 400).filter("lowpass", 15000).filter("loudnorm")
    v = input['v']
    movie = VideoFileClip(i + ".mp4")
    output = ffmpeg.output(a, "tmp_a_from_" + i + ".mp3")
    ffmpeg.run(output)
    output = ffmpeg.output(a, "tmp_voice_opt_from_" + i + ".mp3")
    ffmpeg.run(output)
    #import the new audio
    a = AudioSegment.from_mp3("tmp_a_from_" + i + ".mp3")
    if os.path.exists("tmp_a_from_" + i + ".mp3"):
        os.remove("tmp_a_from_" + i + ".mp3")
    a_voices = AudioSegment.from_mp3("tmp_voice_opt_from_" + i + ".mp3")
    if os.path.exists("tmp_voice_opt_from_" + i + ".mp3"):
        os.remove("tmp_voice_opt_from_" + i + ".mp3")
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
    thresh = mod * max_db
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
    scale_factor = max(movie_height / desired_height, movie_width / desired_width)
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
# add a function that moves any other video file type to a separate folder and creates an mp4 version
def create_video_list(a):
    tmp = []
    for name in os.listdir(a):
        if name.endswith(".mp4"):
            tmp.append(name)
    return tmp

def get_length(filename):
  result = subprocess.Popen(["ffprobe", filename], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines() if "Duration" in x]

print(str(ffmpeg) + " is running")
print("root: " + str(dir))
os.chdir(dir)
vid_arr = create_video_list(dir)
vid_arr.sort(key=lambda x: os.path.getmtime(x))
print("list: " + str(vid_arr) + "")
#base = ffmpeg.input(vid_arr[0])
chunked_clips = []
chunk_folder()
# file_name, max_chunk_size = 10, file_suffix = "default", extention= ".mp3"
chunk_file(vid_arr[0], 10)
sss = ffmpeg.input(chunked_clips[0])
base = process_in_groups(sss, 1.25, 1200, 3)
base_v = base['v']
base_a = base['a']
for b_w in range(len(chunked_clips) - 1):
    sss = ffmpeg.input(chunked_clips[b_w])
    to_concat = process_in_groups(sss, 1.25, 1200, 3)  # 1.4, 15000, 5
    to_concat_v = to_concat['v']
    to_concat_a = to_concat['a']
    print('n = ' + str(0) + ', ' + b_w)
    base = ffmpeg.concat(base_v, base_a, to_concat_v, to_concat_a, v=1, a=1)
    base = base.node
    base_v = base['v']
    base_a = base['a']
if (len(vid_arr) > 1):
    for w in range(1, len(vid_arr) - 1):
        # concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
        chunked_clips = []
        chunk_file(w, 10)
        for c_w in range(len(chunked_clips) - 1):
            sss = ffmpeg.input(chunked_clips[c_w])
            to_concat = process_in_groups(sss, 1.25, 1200, 3) #1.4, 15000, 5
            to_concat_v = to_concat['v']
            to_concat_a = to_concat['a']
            print('n = ' + str(w) + ', ' + c_w)
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
