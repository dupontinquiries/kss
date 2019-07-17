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
import random

from numba import vectorize

t_stamps = {}

#functions

FFMPEG_BIN = 'ffmpeg'
dir = dirname(abspath(__file__)) + "\\footage" or "C:\\Users\\kessl\\Desktop\\Code 2019\\kss\\kss\\footage"
chunked_clips = []
chunked_timestamps = []
clips_to_remove = []

#def val presets
#car voice: 0.96, 500, 1, 0.94, 2560, 1440, 1

#default values
#processing
DEFAULT_THRESHOLD = 0.89
DEFAULT_PERIOD = 3000
DEFAULT_REACH_ITER = 1
DEFAULT_REACH_THRESH = 0.89  # 0.99
DEFAULT_WIDTH = 2560
DEFAULT_HEIGHT = 1440
DEFAULT_MAX_CHUNK_SIZE = .4
verbose = True
cleanup = False
dir = ''
#id
def randomString(stringLength=10):
    string = 'abcdefghijklmnopqrstuvwxyz1234567890'
    """Generate a random string of fixed length """
    letters = string
    return ''.join(random.choice(letters) for i in range(stringLength))
sessionId = randomString()
print(colored('Session ' + sessionId + ' is running...'))
def main(): #call is at the end
    #create actually variables
    THRESHOLD = DEFAULT_THRESHOLD
    PERIOD = DEFAULT_PERIOD
    REACH_ITER = DEFAULT_REACH_ITER
    WIDTH = DEFAULT_WIDTH
    HEIGHT = DEFAULT_HEIGHT
    MAX_CHUNK_SIZE = DEFAULT_MAX_CHUNK_SIZE
    #initialize list for concatenation
    final_cuts = []
    #find directory
    dir = ''
    try:
        if dir is not '':
            dir = 'input from command line'
        else:
            dir = dirname(abspath(__file__)) + '\\footage'
    except:
        print(colored('The directory \"' + dir +'\" was not found!', 'red'))
        sys.exit(0)
    os.chdir(dir)
    if verbose: print(colored('root: ' + str(dir), 'blue'))
    if verbose: print(colored('Finding all video files...', 'blue'))
    vid_arr = create_video_list(dir, False)
    if len(vid_arr) < 1:
        print(colored('No video files were found in \"' + dir + '\"!', 'red'))
        sys.exit(0)
    if verbose: print(colored('Processing files...', 'blue'))
    #gather clips for main file
    for w in range(0, len(vid_arr)):
        print('called vid arr')
    # concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
        process = distr(vid_arr[w], THRESHOLD, PERIOD, REACH_ITER, REACH_ITER, WIDTH, HEIGHT, MAX_CHUNK_SIZE)
        if process != False:
            final_cuts.append(process)
            if verbose:
                print(colored('Addingg file \"' + str(process) + '\" to the final video...', 'blue'))
        else: 
            print(colored('An error was encoutered while adding file #' + str(w) + ' to the final video!', 'red'))
    if verbose: print(colored('Your final video is almost ready...', 'blue'))
    if len(final_cuts) > 1:
        main = moviepy.editor.concatenate_videoclips(final_cuts)
    if len(final_cuts) == 1:
        main = final_cuts[0]
    else:
        print(colored('No clips rendered!', 'red'))
        sys.exit(0)
    if verbose: print(colored('Your final video has a length of ' + main.duration + '!', 'blue'))
    main.write_videofile('final\\final_cut_with_props_' + str(THRESHOLD) + '__' + str(PERIOD) + '__' + str(REACH_ITER) + '__' +  str(WIDTH) + '__' + str(HEIGHT) + '__' + str(MAX_CHUNK_SIZE) + '.mp4')
    #clean up all clips
    if cleanup:
        if verbose: print(colored('Cleaning up the space...', 'blue'))
        for clip in clips_to_remove:
            k_remove(str(clip))
    print(colored('Your video is ready!', 'green'))

def k_xi(p):
    return os.path.exists(p)

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
        name = str("moviepy_subclip_" + str(t_s) + "_" + str(t_f) + "_from_" + str(filename))
        #try
        if not k_xi(name):
            sub = moviepy.video.io.ffmpeg_tools.ffmpeg_extract_subclip(str(filename), t_s, t_f, targetname=name)
            #clips_to_remove.append(name)
        else:
            print('skipping rendering of ' + name)
        t_s += delta
        t_f += delta
        ret = ffmpeg.input(name)
        if not ret:
            yield False
        yield [ret, name]

def k_remove(a):
    if k_xi(a):
        os.remove(a)

#concatenating video files
def k_concat(a):
    if verbose: print(colored('Concatenating a list of files using the k_concat function...', 'blue'))
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

def k_path(p):
    return str(os.path.abspath(str(p)))

#concatenating audio files
def k_map(a, name):
    if verbose: print(colored('Rendering output \'' + name + '\' with the k_map function...', 'blue'))
    b = a[0]
    if len(a) == 0:
        return None
    if len(a) == 1:
        inp = ffmpeg.input(b)
        outp = ffmpeg.output(inp, name)
        inp = VideoFileClip(name)
        return inp
    else:
        if verbose: print(colored('Concatenating list of files...', 'blue'))
        guide_file = 'chunks\\files_to_concat.txt'
        k_remove(guide_file)
        with open(guide_file, "w") as file:
            inputs = ''
            print('a = ' + str(a))
            for x in range(0, len(a)):
                fn = a[x]
                fn = fn.replace('chunks\\', '')
                inputs += 'file \'' + str(fn) + '\'\n'
            if verbose: print(colored('input list: \n' + inputs, 'blue'))
            file.write(inputs)
            # cmd = ('ffmpeg -y ' + inputs + '-filter_complex \'[0:0][1:0][2:0][3:0]concat=n=' + len(a) + ':v=0:a=1[out]\'' + ' -map \'[out]\' ' + name)
            # cmd = ('ffmpeg -y ' + inputs + '-filter_complex \'' + fc + 'concat=n=' + str(len(a) - 1) + ':v=0:a=1[out]\' -map \'[out]\' ' + '\"' + name + '\"')
            # cmd = ('ffmpeg -y ' + inputs + '-filter_complex \'[0:0][1:0][2:0][3:0]concat=n=' + str(len(a)) + ':v=0:a=1[out]\' -map \'[out]\' ' + name)
        #cmd = ('ffmpeg -y -f concat -safe 0 -i \"' + k_path(guide_file) + '\" -c copy ' + '\"' + k_path(name) + '\"')
        cmd = ('ffmpeg -y -f concat -safe 0 -i \"' + k_path(guide_file) + '\" ' + '\"' + k_path(name) + '\"')
        print('cmd = ' + cmd)
        subprocess.call(cmd)
        if verbose: print(colored('Importing resulting file...', 'blue'))
        inp = VideoFileClip(name)
        clips_to_remove.append(guide_file)
        clips_to_remove.append(name)
        return inp

def mpy_concat(filenames, output_name):
    if verbose: print(colored('Concatenating a list of files using the mpy_concat function...', 'blue'))
    mpys_v = []
    mpys_a = []
    for fn in enumerate(filenames):
        _v = VideoFileClip(fn)
        _a = _v.audio
        mpys_v.append(_v)
        mpys_a.append(_a)
        del _v
        del _a
    v_t = moviepy.editor.concatenate_videoclips(mpys_v)
    a_t = moviepy.editor.concatenate_videoclips(mpys_a)
    v_t.set_audio(a_t)
    v_t.write_videofile(output_name)

def distr(filename, mod, c_l, spread, thresh_mod, crop_w, crop_h, max_chunk_size):
    base_name = filename[:-4]
    # compress any large files
    smaller_clips = []
    print("attempting to distr() " + filename)
    if verbose: print(colored('Verifying clip...', 'blue'))
    if "completed" in filename or "output_from_all" in filename or "sublcip" in filename or "moviepy" in filename:
        return False
    tmp_clip = False
    #get duration
    try:
        if verbose: print(colored('Finding length...', 'blue'))
        tmp_clip = VideoFileClip(filename)
        l = tmp_clip.duration
        tmp_clip.close()
        del tmp_clip
    except:
        console_a = ''
        if not k_xi(filename):
            console_a = 'not'
        print(colored('An error was encountered while opening \"' + filename+ '\"!  The file seems to ' + console_a + ' exist.', 'red'))
        sys.exit(0)
    if verbose: print(colored('Chunking clip...', 'blue'))
    for piece in read_in_ffmpeg_chunks(filename, max_chunk_size, l):
        if piece is not False:
            if file_size(filename) >= (10 ** 9):
                if verbose: print("file " + str(filename) + " is large (" + str(file_size(filename)) + ").  (Future Capability) Keeping the chunked clips as \"cc\"")
            print('piece: ' + str(piece))
            result = process_audio_loudness_over_time(piece[0], piece[1], mod, c_l, spread, thresh_mod, crop_w, crop_h)
            if result is not False:
                smaller_clips.append(result)
    if len(smaller_clips) >= 1:
        output_name = "final\\completed_file_" + base_name
        if verbose: print(colored('Writing clip \'' + output_name + '.mp4' + '\' from smaller clips...', 'blue'))
        return k_map(smaller_clips, output_name + '.mp4')
    else:
        return False

def render_(component):
    ffmpeg.run(component, overwrite_output=True)

def floor_out(a, bottom):
    if a < bottom:
        return bottom
    else:
        return a

def k_f2v(c_l, movie_a_fc, a_voices_fc, spread, mod_solo, mod_multi, duration, movie_v, movie_a, concat_log):
    tc_v = []
    tc_a = []
    with open(concat_log, "r") as file:
        for line in file:
            line = line.replace('subclip', '')
            start = float(line.split('t')[0])
            cap = float(line.split('t')[1])
            tmp_v = movie_v.subclip(start, cap)
            tc_v.append(tmp_v)
            tmp_a = movie_a.subclip(start, cap)
            tc_a.append(tmp_a)
            tmp_v.close()
            del tmp_v.reader
            del tmp_v
            del tmp_a
            print(colored('Section ' + str(start) + ' to ' + str(cap), 'green'))
    if len(tc_v) < 1:
        print(colored("No packets came through.", 'red'))
        return False
    processed_v = None
    processed_a = None
    if len(tc_v) > 1:
        if verbose: print(colored('Concatenating snippets...', 'blue'))
        if verbose: print(colored('Concatenating a list of files from a chunk...', 'blue'))
        processed_a = moviepy.editor.concatenate_audioclips(tc_a)
        processed_v = moviepy.editor.concatenate_videoclips(tc_v)
        print(colored("Concatenating accepted packets.", 'green'))
    else:
        processed_v = tc_v[0]
        processed_a = tc_a[0]
        if verbose: print(colored(
            'Only one packet made it through the vetting process... you should consider lowering your threshold value or changing the search distance value.',
            'blue'))
    return [processed_v, processed_a]

def k_filter_loudness(c_l, movie_a_fc, a_voices_fc, spread, mod_solo, mod_multi, duration, movie_v, movie_a, concat_log):
    # create averaged audio sections and decide which ones meet threshold
    if verbose: print(colored('Chunking audio...', 'blue'))
    chunk_length_ms = c_l
    chunk_length_s = chunk_length_ms / 1000
    # maybe use iter_chunks
    chunks_a = make_chunks(movie_a_fc, chunk_length_ms)
    chunks_a_voice = make_chunks(a_voices_fc, chunk_length_ms)
    tc_v = []
    tc_a = []
    list_of_db = []
    list_of_db_solo = []
    spread_calc = int(((spread - 1) / 2))
    # get start
    db_arr = 0
    if verbose: print(colored('Creating snippets...', 'blue'))
    for q_1 in range(0, 2 * spread_calc):
        db_arr += chunks_a_voice[q_1].dBFS
    db = db_arr / spread
    for z_init in range(0, spread_calc):  # removed -1
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
    # reformat the sound levels
    if verbose: print(colored('Flooring audio to remove -inf sections...', 'blue'))
    floor = 150
    if verbose: print(colored('Floor = ' + str(floor), 'blue'))
    list_of_db = list(map(lambda x: floor_out(x, - floor), list_of_db))
    list_of_db_solo = list(map(lambda x: floor_out(x, - floor), list_of_db_solo))
    list_of_db = list(map(lambda x: x + floor, list_of_db))
    list_of_db_solo = list(map(lambda x: x + floor, list_of_db_solo))
    # get target threshold to use for modifiers
    max_db = max(list_of_db)
    median_db = statistics.median(list_of_db)
    average_db = statistics.mean(list_of_db)
    max_db_solo = max(list_of_db_solo)
    median_db_solo = statistics.median(list_of_db_solo)
    average_db_solo = statistics.mean(list_of_db_solo)
    target_db = ((.5 * median_db_solo) + (.4 * average_db_solo) + (.1 * max_db_solo))
    thresh = mod_solo * target_db
    target_db = ((.7 * median_db) + (.2 * average_db) + (.1 * max_db))
    thresh_multi = mod_multi * target_db
    # logging purposes
    # print("max_db_solo: " + str(max_db_solo) + "/" + floor)
    # print("max_db_multi: " + str(max_db) + "/" + floor)
    if verbose: print(colored('thresh_solo = ' + str(thresh) + '\nthresh_multi = ' + str(thresh_multi), 'blue'))
    if verbose: print(colored('Vetting snippets...', 'blue'))
    inputs = ''
    if len(chunks_a) > 1:
        for x in range(0, len(chunks_a) - 1):
            # op1 - harsh analysis on long pieces
            raw = list_of_db[x]  # group
            raw_solo = list_of_db_solo[x]  # group
            # if raw_solo > thresh or raw > thresh_multi:
            if raw_solo > thresh or raw > thresh_multi:
                start = max(0, x * chunk_length_s)
                cap = (x + 1) * chunk_length_s  # min((x + 1) * chunk_length_s, duration)
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
                    inputs += 'subclip' + str(start) + 't' + str(cap) + '\n'
                    print(colored("Section #" + str(x)
                                  + "\n" + "peak volume = " + str(raw_solo)
                                  + "\n" + "avg  volume = " + str(raw), 'green'))
                else:
                    print(colored("Failed to render this subclip due to errors with the time parameters!", 'red'))
            else:
                print(colored("Section #" + str(x)
                              + "\n" + "peak volume = " + str(raw_solo)
                              + "\n" + "avg  volume = " + str(raw), 'red'))
        # combine all clips into one
    else:
        print(colored('Error creating chunks of audio!  Not enough chunks created.', 'red'))
        return False
    if len(tc_v) < 1:
        print(colored("No packets came through.", 'red'))
        return False
    processed_v = None
    processed_a = None
    if len(tc_v) > 1:
        if verbose: print(colored('Concatenating snippets...', 'blue'))
        if verbose: print(colored('Concatenating a list of files from a chunk...', 'blue'))
        processed_a = moviepy.editor.concatenate_audioclips(tc_a)
        processed_v = moviepy.editor.concatenate_videoclips(tc_v)
        print(colored("Concatenating accepted packets.", 'green'))
    else:
        processed_v = tc_v[0]
        processed_a = tc_a[0]
        if verbose: print(colored(
            'Only one packet made it through the vetting process... you should consider lowering your threshold value or changing the search distance value.',
            'blue'))
    with open(concat_log, "w") as file:
        file.write(inputs)
    return [processed_v, processed_a]

#merge_outputs = combine clips; overwrite_output = overwrite files /save lines of code
def process_audio_loudness_over_time(i, name, mod_solo, c_l, spread, mod_multi, crop_w, crop_h):
    #create log for future renderings
    concat_log = 'chunks\\concat_log_args_' + str(name) + '__' + str(mod_solo) + '__' + str(c_l) + '__' + str(spread) + '__' + str(mod_multi) + '__' + str(crop_w) + '__' + str(crop_h) + '.txt '
    #get root of file name
    og = name
    name = str(name.replace(".mp4", ""))
    input = i
    #audio
    name_audio = 'chunks\\tmp_a_from_' + name + '.wav'
    name_audio_voice = 'chunks\\tmp_voice_opt_from_' + name + '.wav'
    if verbose: print(colored('Preparing audio for video...', 'blue'))
    #video clip audio
    a = None
    if not k_xi(name_audio):
        a = input['a']
        # clean up audio so program takes loudness of voice into account moreso than other sounds
        # clean up audio of final video
        if verbose: print(colored('Preparing tailored audio...', 'blue'))
        a = a.filter('highpass', 35).filter("lowpass", 18000).filter("loudnorm")
        # export clip audio
        if verbose: print(colored('Writing tailored audio...', 'blue'))
        output = ffmpeg.output(a, name_audio)
        render_(output)
    if verbose: print(colored('Importing tailored audio from \"' + name_audio + '\"...', 'blue'))
    a = ffmpeg.input(name_audio)
    #voice_opt
    a_voice = None
    if not k_xi(name_audio_voice):
        if verbose: print(colored('Preparing audio for analysis...', 'blue'))
        a_voice = a.filter("afftdn", nr=16, nt="w", om="o").filter('highpass', 200).filter("lowpass", 8000).filter("loudnorm")
        # export voice_optimized audio
        output = ffmpeg.output(a_voice, name_audio_voice)
        render_(output)
    if verbose: print(colored('Importing optimized audio from \"' + name_audio_voice + '\"...', 'blue'))
    a_voice = ffmpeg.input(name_audio_voice)
    #import the new voice_opt audio
    if verbose: print(colored('Establishing audio files...', 'blue'))
    movie_a_fc = AudioSegment.from_mp3(name_audio)
    a_voices_fc = AudioSegment.from_mp3(name_audio_voice)
    movie_a = AudioFileClip(name_audio)
    a_voices = AudioFileClip(name_audio_voice)
    #add them to delete list
    clips_to_remove.append(name_audio)
    clips_to_remove.append(name_audio_voice)
    #get subclips in the processing part
    if verbose: print(colored('Opening clip \'' + og + '\'...', 'blue'))
    movie_v = VideoFileClip(og)
    duration = None
    try:
        duration = movie_v.duration
    except:
        print(colored('Failed to open clip \'' + og + '\' and find its length!', 'red'))
        return False
    if k_xi(concat_log):
        if verbose: print(colored('Found documentation of what clips to use...', 'blue'))
        ret_info = k_f2v(c_l, movie_a_fc, a_voices_fc, spread, mod_solo, mod_multi, duration, movie_v, movie_a, concat_log)
        processed_v = ret_info[0]
        processed_a = ret_info[1]
    else:
        if verbose: print(colored('No documentation found: creating clips and documentation from scratch...', 'blue'))
        ret_info = k_filter_loudness(c_l, movie_a_fc, a_voices_fc, spread, mod_solo, mod_multi, duration, movie_v, movie_a, concat_log)
        processed_v = ret_info[0]
        processed_a = ret_info[1]
    #export clip
    if verbose: print(colored('Combining video and audio...', 'blue'))
    processed_v = processed_v.set_audio(processed_a)
    duration = processed_v.duration
    if verbose: print(colored('Writing new files from merged snippets...', 'blue'))
    base_name = 'chunks\\processed_output_from_' + name
    processed_v.write_videofile(base_name + '.mp4')
    processed_a.write_audiofile(base_name + '.wav')
    clips_to_remove.append(base_name + '.mp4')
    clips_to_remove.append(base_name + '.wav')
    #reopen combined clip
    if verbose: print(colored('reopening ' + base_name + '.mp4' + ' using ffmpeg...', 'blue'))
    #reimport for filtering
    # filter video
    ret = ffmpeg.input(base_name + '.mp4')
    movie_height = movie_v.h
    desired_height = crop_h
    movie_width = movie_v.w
    desired_width = crop_w
    scale_factor = min((movie_height / desired_height), (movie_width / desired_width))
    if verbose: print(colored('Resizing video with a scale factor of ' + str(scale_factor) + ', and dimentions w: ' + str(desired_width) + ' and h: ' + str(desired_height) + '...', 'blue'))
    base_v = ret.video.filter('crop', x=1.50*crop_w*scale_factor, y=0*crop_h*scale_factor, w=crop_w*scale_factor, h=crop_h*scale_factor)
    base_a = ret.audio.filter("loudnorm").filter("afftdn", nr=8, nt="w", om="o").filter("equalizer", f=7000,
                                                                                        width_type="o", width=5,
                                                                                        g=1).filter("equalizer", f=200,
                                                                                                    width_type="o",
                                                                                                    width=2,
                                                                                                    g=-1)
    output_file = 'chunks\\filtered_and_processed_output_from_' + name
    if verbose: print(colored('Writing filtered files...', 'blue'))
    output_v = ffmpeg.output(base_v, output_file + '.mp4')
    output_a = ffmpeg.output(base_a, output_file + '.wav')
    render_(output_v)
    render_(output_a)
    processed_v = VideoFileClip(output_file + '.mp4')
    processed_a = AudioFileClip(output_file + '.wav')
    processed_v.set_audio(processed_a)
    processed_v.write_videofile(output_file + '.mp4')
    clips_to_remove.append(output_file + '.mp4')
    clips_to_remove.append(output_file + '.wav')
    return output_file + '.mp4'

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
        if name.endswith(".m2ts") or name.endswith(".mov"):
            to_mp4(name)
            os.rename(name, 'not_mp4\\' + name)
    for name in os.listdir(a):
        if name.lower()[-4:] == '.mp4':
            name = name[:-4] + '.mp4'
            tmp.append(name)
            if ts:
                create_timestamps(name)
    return tmp

def get_length(filename):
  result = subprocess.Popen(["ffprobe", filename], stdout = subprocess.PIPE, stderr = subprocess.STDOUT)
  return [x for x in result.stdout.readlines() if "Duration" in x]


#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
#with tf.device("/cpu:0"):
#print("current device = " + str(torch.cuda.current_device()))
#with torch.cuda.device(1):
main()