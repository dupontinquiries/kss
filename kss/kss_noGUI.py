## TODO: add folder selection and auto folder creation


#imports

from __future__ import unicode_literals

#av

#video

import ffmpeg
from moviepy.video.io.ffmpeg_tools import *
import moviepy
import moviepy.editor as mpye

#audio

import cv2
from pydub import AudioSegment
from pydub.utils import make_chunks

#math

import statistics
import random

#readability

from termcolor import colored

#fs

import os
from os.path import dirname, abspath
import sys
import shutil
import subprocess

#my stuff

from k_chunk import k_chunk

# import torch as torch

# from numba import vectorize

#store user input

inputLog = []
curr_input = None

t_stamps = {}

# functions

FFMPEG_BIN = 'ffmpeg'
dir = dirname(abspath(__file__)) + "\\footage"
chunked_clips = []
chunked_timestamps = []
clips_to_remove = []

# def val presets
# car voice: 0.96, 500, 1, 0.94, 2560, 1440, 1

# default values
# processing
DEFAULT_THRESHOLD = 1.5
DEFAULT_PERIOD = 350
DEFAULT_REACH_ITER = 10
DEFAULT_REACH_THRESH = .9 * DEFAULT_THRESHOLD
DEFAULT_WIDTH = 1920  # 2560
DEFAULT_HEIGHT = 1080  # 1440
DEFAULT_MAX_CHUNK_SIZE = 10 #1.2, 3.2, 10.2
DEFAULT_TREATMENT = list(['voice', 'music'])[0]
verbose = True
cleanup = False
dir = ''
print_color = 'cyan'
print_color_error = 'yellow'


# id / random string generator
def randomString(stringLength=10):
    string = 'abcdefghijklmnopqrstuvwxyz1234567890'
    """Generate a random string of fixed length """
    letters = string
    return ''.join(random.choice(letters) for i in range(stringLength))


sessionId = randomString()

print(colored('Session ' + sessionId + ' is running...'))


def main():  # call is at the end
    # create actual variables
    THRESHOLD = DEFAULT_THRESHOLD
    REACH_THRESH = DEFAULT_REACH_THRESH
    PERIOD = DEFAULT_PERIOD
    REACH_ITER = DEFAULT_REACH_ITER
    WIDTH = DEFAULT_WIDTH
    HEIGHT = DEFAULT_HEIGHT
    MAX_CHUNK_SIZE = DEFAULT_MAX_CHUNK_SIZE
    # initialize list for concatenation
    final_cuts = []
    # find directory
    dir = ''
    try:
        if dir is not '':
            dir = 'input from command line'
        else:
            dir = dirname(abspath(__file__)) + '\\footage'
    except:
        print(colored('The directory \"' + dir + '\" was not found!', print_color_error))
        sys.exit(0)
    os.chdir(dir)
    print('dir = {0}'.format(os.getcwd()))
    if verbose: print(colored('root: ' + str(dir), print_color))
    if verbose: print(colored('Finding all video files...', print_color))
    #print(os.path.abspath(dir + '\\input'))
    #exit()
    subfolder = 'input'
    vid_arr = create_video_list(dir, False)  # dir, create time stamps and reorder based on time blocks, time block duration
    if len(vid_arr) < 1:
        print(colored('No video files were found in \"' + dir + '\"!', print_color_error))
        sys.exit(0)
    if verbose: print(colored('Processing files...', print_color))
    # gather clips for main file
    maxi = max(1, len(vid_arr) - 1)
    for w in range(0, maxi):
        # concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
        process = distr(vid_arr[w], subfolder, THRESHOLD, PERIOD, REACH_ITER, REACH_ITER, WIDTH, HEIGHT, MAX_CHUNK_SIZE)
        if process != False:
            final_cuts.append(process)
            if verbose:
                print(colored('Adding file \"' + str(process) + '\" to the final video...', print_color))
        else:
            print(colored('An error was encoutered while adding file #' + str(w) + ' to the final video!', print_color_error))
    if verbose: print(colored('Your final video is almost ready...', print_color))
    main = None
    if len(final_cuts) == 1:
        main = final_cuts[0]
    elif len(final_cuts) > 1:
        main = mpye.concatenate_videoclips(final_cuts)
    else:
        print(colored('No clips rendered!', print_color_error))
        sys.exit(0)
    if verbose: print(colored('Your final video has a length of ' + str(main.duration) + '!', print_color))
    final_name = 'final\\final_cut_with_props_{0}_{1}_{2}_{3}_{4}_{5}' \
        .format(THRESHOLD, REACH_THRESH, PERIOD, REACH_ITER, WIDTH, HEIGHT) \
        .replace('.', '-') + '.mp4'
    #print('final video filename = {0}'.format(final_name))
    print(main.duration)
    main.write_videofile(final_name)
    # clean up all clips
    if cleanup:
        if verbose: print(colored('Cleaning up the space...', print_color))
        for clip in clips_to_remove:
            k_remove(str(clip))
    print(colored('Your video is ready!', 'green'))


def kIn(x):
    b = input(x)
    curr_input = b
    print(b)
    return b


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


def kFileCharacteristic(filename, ret='all'):
    cmd = 'ffprobe "{0}" -show_format ' \
        .format(filename)
    result = subprocess \
        .Popen(cmd, \
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    results = result.communicate()
    w = None
    if ret == 'all':
        w = results
    else:
        w = float(str(results).split(ret)[1].split('\\')[0].replace('=', ''))
    #print(results)
    result.kill()
    return w


def read_in_ffmpeg_chunks(filename, max_chunk_size):
    file_length = 0
    if verbose:
        print(colored('Finding length...', print_color))
    file_length = kFileCharacteristic(filename, 'nduration')
    max_chunk_size *= 60  # convert to seconds
    t_s = 0
    t_f = min(file_length, max_chunk_size)
    if file_length < max_chunk_size:
        try:
            ret_video = ffmpeg.input(name)
            ret_audio = ret_video.audio
        except:
            return False
        return (ret_video, ret_audio, name)
    # itereate through the file and make chunks
    while file_length - t_s > 0:
        delta = t_f - t_s
        if verbose:
            print('t_s = {0}, t_f = {1}, d = {2}'.format(t_s, t_f, delta))
        if file_length - t_f <= 0:
            yield False
        name = 'moviepy_subclip_{0}_{1}_from_{2}'\
            .format(t_s, t_f, filename)
        rootName = name
        # generates a subclip to avoid memory caps
        # saves time on reruns to skip if the file already exists
        if not k_xi(name):
            # export subclip
            cmd = 'ffmpeg -y -i "{0}" -ss {1} -t {2} -c:v copy -c:a copy -strict 1 "{3}"'\
                .format(filename, t_s, min(delta, file_length - t_s), name)
            print('[cmd] ~ {0}'.format(cmd))
            os.system(cmd)
            #subprocess.run(cmd)
            #cmd.kill()
            clips_to_remove.append(name)
        else:
            print('skipping chunk-rendering of clip \"' + name + '\"')
        #make an audio import to map streams better
        if not k_xi(rootName + '.mp3'):
            cmd = 'ffmpeg -y -i "{0}" "{1}"'\
                .format(name[:-4] + '.mp4', name[:-4] + '.mp3')
            print('[cmd] ~ {0}'.format(cmd))
            os.system(cmd)
        #subprocess.run(cmd)
        #cmd.kill()
        clips_to_remove.append(name)
        #update time
        t_s += delta
        t_f += delta
        # ret = VideoFileClip(name)
        # print('nav = ' + str(ret))
        try:
            ret_video = ffmpeg.input(name)
            ret_audio = ffmpeg.input(name[:-4] + '.mp3')
        except:
            yield False
        if ret_video == None or ret_audio == None:
            yield False
        yield (ret_video, ret_audio, name)


def k_remove(a):
    if k_xi(a):
        os.remove(a)


# concatenating video files
def k_concat(a):
    if verbose: print(colored('Concatenating a list of files using the k_concat function...', print_color))
    b = a[0]
    if len(a) == 0:
        return None
    if len(a) == 1:
        return b
    else:
        for x in range(1, len(a) - 1):
            c = a[x]
            b = ffmpeg.concat(b['v'], b['a'], c['v'], c['a'], v=1, a=1)
        return b


def k_path(p):
    return str(os.path.abspath(str(p)))


# concatenating audio files
def k_map(a, name):
    if verbose: print(colored('Rendering output \'' + name + '\' with the k_map function...', print_color))
    b = a[0]
    if len(a) == 0:
        return None
    if len(a) == 1:
        inp = mpye.VideoFileClip(b)
        try:
            inp.close()
        except:
            if verbose:
                print('failed to close input = {0}'.format(name))
        return inp
    else:
        if verbose: print(colored('Concatenating list of files...', print_color))
        guide_file = 'chunks\\files_to_concat.txt'
        k_remove(guide_file)
        with open(guide_file, "w") as file:
            inputs = ''
            print('a = ' + str(a))
            for x in range(0, len(a) - 1):
                fn = a[x]
                fn = fn.replace('chunks\\', '')
                inputs += 'file \'' + str(fn) + '\'\n'
            if verbose: print(colored('input list: \n' + inputs, print_color))
            file.write(inputs.strip())
            #-fs 1GB
        cmd = 'ffmpeg -y -f concat -safe 0 -i "{0}" -c:v copy -c:a copy -strict 1 "{1}"' \
            .format(k_path(guide_file), k_path(name))
        print('cmd = ' + cmd)
        subprocess.call(cmd)
        if verbose: print(colored('Importing resulting file...', print_color))
        inp = mpye.VideoFileClip(name)
        try:
            inp.close()
        except:
            if verbose:
                print('failed to close input = {0}'.format(name))
        clips_to_remove.append(guide_file)
        clips_to_remove.append(name)
        return inp


def mpy_concat(filenames, output_name):
    if verbose: print(colored('Concatenating a list of files using the mpy_concat function...', print_color))
    mpys_v = []
    mpys_a = []
    for fn in enumerate(filenames):
        _v = VideoFileClip(fn)
        _a = _v.audio
        mpys_v.append(_v)
        mpys_a.append(_a)
        del _v
        del _a
    v_t = concatenate_audioclips(mpys_v)
    a_t = concatenate_videoclips(mpys_a)
    v_t.set_audio(a_t)
    v_t.write_videofile(output_name)

def distr(filename, subfolder, mod, c_l, spread, thresh_mod, crop_w, crop_h, max_chunk_size):
    base_name = filename[:-4].replace(subfolder, '')
    # compress any large files
    smaller_clips = []
    if verbose: print(colored('Verifying clip \"' + filename + '\"', print_color))
    l_ignore = ['completed', 'output_from_all', 'subclip', 'moviepy']
    for o in l_ignore:
        if o in filename:
            return False
    tmp_clip = False
    # get duration
    if verbose:
        print(colored('Finding length...', print_color))
    tmp_clip = mpye.VideoFileClip(filename)
    l = tmp_clip.duration
    try:
        tmp_clip.close()
        del tmp_clip
    except:
        exist_condition = k_xi(filename)
        print(colored('An error was encountered while closing the reader for \"{0}\" ({1})' \
            .format(filename, exist_condition), print_color_error))
    if not l > 0:
        exist_condition = k_xi(filename)
        print(colored('An error was encountered while closing the reader for \"{0}\" ({1})' \
            .format(filename, exist_condition), print_color_error))
        sys.exit(0)
    if verbose: print(colored('Chunking clip...', print_color))
    for piece in read_in_ffmpeg_chunks(filename, max_chunk_size):
        if piece is not False:
            input_video = piece[0]
            input_audio = piece[1]
            file_name_chunk = piece[2]
            if verbose:
                print('Opening chunk \"{0}\"'.format(file_name_chunk))
            result = process_audio_loudness_over_time(input_video, input_audio, file_name_chunk, mod, c_l, spread, thresh_mod, crop_w, crop_h)
            if result is not False:
                smaller_clips.append(result)
                if verbose: print(colored('Adding clip "{0}" to the list...' \
                    .format(file_name_chunk), print_color))
            else:
                print(colored('No data appended for chunk "{0}"' \
                    .format(file_name_chunk), print_color))
    output_name = "final\\completed_file_" + base_name
    if len(smaller_clips) == 1:
        if not k_xi(output_name + '.mp4'):
            shutil.copy(smaller_clips[0], output_name + '.mp4')
        return mpye.VideoFileClip(output_name + '.mp4')
    elif len(smaller_clips) > 1:
        if verbose:
            print(colored('Writing clip \'' + output_name + '.mp4' + '\' from smaller clips...', print_color))
        c_tmp = []
        for o in smaller_clips:
            c_tmp.append(mpye.VideoFileClip(o))
        return mpye.concatenate_videoclips(c_tmp)
        #return k_map(smaller_clips, output_name + '.mp4')
    else:
        return False


def render_(component):
    ffmpeg.run(component, overwrite_output=True)


def floor_out(a, bottom):
    if a < bottom:
        return bottom
    else:
        return a


def k_round(i, d):
    n = 1
    for a in range(0, d):
        n *= 10
    return int(n * i) / n


def k_tf(seconds):
    hours = seconds // (60 * 60)
    seconds %= (60 * 60)
    minutes = seconds // 60
    seconds %= 60
    print('seconds pre = ' + str(seconds))
    print('seconds post = ' + str(seconds))
    ret = "%02i:%02i:%02i" % (hours, minutes, seconds)
    ret += str(seconds - int(seconds))[1:]
    print('format = ' + ret)
    return ret


def k_partition(sort_list, low, high, fn):
    i = (low - 1)
    pivot = sort_list[high]
    for j in range(low, high):
        if fn(sort_list[j]) <= fn(pivot):
            i += 1
            sort_list[i], sort_list[j] = sort_list[j], sort_list[i]
    sort_list[i + 1], sort_list[high] = sort_list[high], sort_list[i + 1]
    return (i + 1)


def k_quick_sort(sort_list, low, high, fn):
    if low < high:
        pi = k_partition(sort_list, low, high, fn)
        k_quick_sort(sort_list, low, pi - 1, fn)
        k_quick_sort(sort_list, pi + 1, high, fn)


def get_sv(e):
    return e.sv


def get_v(e):
    return e.v


def get_ts(e):
    return e.t_s


def k_splval(l, tr, fn):
    tmp = l.copy()
    n = len(l) - 1
    while fn[0](tmp[n]) > tr[0] and n > 0:
        #and fn[1](tmp[n]) > tr[1] \
        n -= 1
    return tmp[n+1:]


def lSum(l, fn):
    sum = 0
    for o in l:
        sum += fn(o)
    return sum


def medianOf(l, fn):
    l2 = []
    for o in l:
        l2.append(fn(o))
    return statistics.median(l2)


def k_stats(cl):
    from datetime import datetime
    # stats
    print('len(kstats) = {0}'.format(len(cl)))
    f_spread = cl.copy()
    start = datetime.now()
    k_quick_sort(f_spread, 0, len(f_spread) - 1, get_sv)
    end = datetime.now()
    if verbose: print('time to sort by spread values: {0}'.format(end - start))
    f_solo = cl.copy()
    start = datetime.now()
    k_quick_sort(f_solo, 0, len(f_solo) - 1, get_v)
    end = datetime.now()
    if verbose: print('time to sort by chunk volume: {0}'.format(end - start))

    #avg
    start = datetime.now()
    sums = (lSum(cl, get_v), lSum(cl, get_sv))
    ln = len(cl)
    if ln < 1:
        return False
    averages = (sums[0]/ln, sums[1]/ln)
    end = datetime.now()
    if verbose: print('time to generate averages: {0}'.format(end - start))
    if verbose: print('averages = {0}'.format(averages))

    #median
    start = datetime.now()
    medians = (f_solo[min(len(f_solo), len(f_solo) // 2)], \
        f_spread[min(len(f_spread), len(f_spread) // 2)])
    end = datetime.now()
    if verbose: print('time to generate medians: {0}'.format(end - start))
    if verbose: print('medians = {0}'.format(medians))

    c_medians = (medianOf(cl, get_v), medianOf(cl, get_sv))
    if verbose: print('calculated medians = {0}'.format(c_medians))

    start = datetime.now()
    maxes = (f_solo[-1], f_spread[-1])
    end = datetime.now()
    if verbose: print('time to generate maxes: {0}'.format(end - start))
    if verbose: print('maxes = {0}'.format(maxes))

    start = datetime.now()
    w = k_chunk(0, None, 0, 0, 0, 0, 'nofile', True).DEFAULT_FLOOR
    thresholds = \
    (DEFAULT_THRESHOLD * ((averages[0] * .15) + (medians[0].v * .25) + (maxes[0].v * .6)), \
    DEFAULT_REACH_THRESH * ((averages[1] * .15) + (medians[1].sv * .25) + (maxes[1].sv * .6)))

    if False:
        thresholds = \
        (DEFAULT_THRESHOLD * ((averages[0] * .15) + (medians[0].v * .15) + (maxes[0].v * .3)), \
        DEFAULT_REACH_THRESH * ((averages[1] * .15) + (medians[1].sv * .15) + (maxes[1].sv * .3)))
    end = datetime.now()
    if verbose: print('time to generate thresholds: {0}'.format(end - start))
    if verbose: print('thresholds = {0}'.format(thresholds))

    start = datetime.now()
    rem_solo = k_splval(f_solo.copy(), thresholds, (get_v, get_sv))
    rem_spread = k_splval(f_spread.copy(), thresholds, (get_v, get_sv)) #lists should be the same
    end = datetime.now()
    if verbose:
        print('time to generate trimmed lists: {0}'.format(end - start))
    if verbose:
        print('len(solo) = {0}\nlen(rem_solo) = {1}\npercentage = {2:.3f}' \
            .format(len(f_solo), len(rem_solo), len(rem_solo) / len(f_solo)))
    if verbose:
        print('len(spread) = {0}\nlen(rem_spread) = {1}\npercentage = {2:.3f}' \
            .format(len(f_spread), len(rem_spread), len(rem_spread) / len(f_spread)))

    avgP = .5 * (len(rem_solo) / len(f_solo)) + .5 * (len(rem_spread) / len(f_spread))

    k_quick_sort(rem_solo, 0, len(rem_solo) - 1, get_ts)
    k_quick_sort(rem_spread, 0, len(rem_spread) - 1, get_ts)

    #if verbose: print('ordered solo list:\n{0}\nordered spread list:\n{1}' \
    #    .format(rem_solo, rem_spread))

    #make the golden list: a blend of spread and solo values that focuses on making cohesive cuts
    goldenList = []
    for o in cl:
        #if o in rem_solo or o in rem_spread:
        if o.v > thresholds[0] or o.sv > thresholds[1]:
            goldenList.append(o)

    print('{0}'.format(thresholds))
    print('{0}, {1}, {2}'.format(len(goldenList), len(rem_solo), len(rem_spread)))
    #print(goldenList)


    # TODO:
    # make dictionary of values anmd pass nack to parent function
    # add recursive tuning loop that can adjust threshold value by abs(.1 * floor) value

    return (rem_solo, thresholds[0], rem_spread, thresholds[1], goldenList, avgP)

def wrapper(func, *args):
    func(*args)

# merge_outputs = combine clips; overwrite_output = overwrite files /save lines of code
def process_audio_loudness_over_time(input_video, input_audio, name, mod_solo, c_l, spread, mod_multi, crop_w, crop_h):
    # create log for future renderings
    concat_log = 'chunks\\c_l_{0}_{1:.1f}_{2:.1f}_{3:.1f}_{4:.1f}_{5:.1f}_{6:.1f}.txt' \
        .format(name, mod_solo, c_l, spread, mod_multi, crop_w, crop_h)
    # get root of file name
    og = name
    name = name[:-4]
    #check if need to render anything
    base_name = 'fclips\\processed_output_from_' + name
    output_file = 'flcips\\filtered_and_processed_output_from_' + name
    if k_xi(output_file):
        preint('tecca')
        return output_file
    # audio
    name_audio = 'chunks\\tmp_a_from_' + name + '.mp3'
    name_audio_voice = 'chunks\\tmp_voice_opt_from_' + name + '.mp3'
    if verbose: print(colored('Preparing audio for video...', print_color))
    # video clip audio
    #set values for treatment type
    highpass = 80
    lowpass = 1200
    if DEFAULT_TREATMENT == 'music':
        highpass = 0
        lowpass = 20000
    if not k_xi(name_audio):

        a_name_audio = input_audio \
            .filter("loudnorm")
            #.filter("afftdn", nr=6, nt="w", om="o") \
            #.filter("afftdn", nr=2, nt="w", om="o") \
        # clean up audio so program takes loudness of voice into account moreso than other sounds
        # clean up audio of final video
        if verbose: print(colored('Preparing tailored audio...', print_color))
        # export clip audio

        output = ffmpeg.output(a_name_audio, name_audio)
        if verbose: print(colored('Writing tailored audio...', print_color))
        render_(output)
    if verbose: print(colored('Importing tailored audio from \"' + name_audio + '\"...', print_color))
    a = ffmpeg.input(name_audio)
    # voice_opt
    if not k_xi(name_audio_voice):
        if verbose: print(colored('Preparing audio for analysis...', print_color))
        a_voice = a \
            .filter("afftdn", nr=12, nt="w", om="o") \
            .filter('highpass', highpass) \
            .filter("lowpass", lowpass) \
            .filter("afftdn", nr=12, nt="w", om="o") \
            .filter("loudnorm")
        # export voice_optimized audio
        output = ffmpeg.output(a_voice, name_audio_voice)
        render_(output)
    if verbose: print(colored('Importing optimized audio from \"' + name_audio_voice + '\"...', print_color))
    a_voice = ffmpeg.input(name_audio_voice)
    # import the new voice_opt audio
    if verbose: print(colored('Establishing audio files...', print_color))
    movie_a_fc = AudioSegment.from_mp3(name_audio)
    a_voices_fc = AudioSegment.from_mp3(name_audio_voice)
    if verbose: print(colored('Name of audio file is \"' + name_audio + '\"', print_color))


    movie_a = mpye.AudioFileClip(name_audio)
    try:
        movie_a.close()
        movie_a.reader.close()
    except:
        if verbose: print(colored('Error closing reader of optimized audio from \"' + name_audio_voice + '\"...', print_color_error))
    #print(movie_a)

    movie_a_length = movie_a.duration
    a_voices = mpye.AudioFileClip(name_audio_voice)
    try:
        a_voices.close()
        a_voices.reader.close()
    except:
        if verbose: print(colored('Error closing reader of optimized audio from \"' + name_audio_voice + '\"...', print_color_error))
    # add them to delete list
    clips_to_remove.append(name_audio)
    clips_to_remove.append(name_audio_voice)
    # get subclips in the processing part
    if verbose: print(colored('Opening clip \'' + og + '\'...', print_color))
    movie_v = mpye.VideoFileClip(og)
    try:
        movie_v.close()
    except:
        if verbose:
            print('failed to close movie_v = {0}'.format(og))
    # test_input = ffmpeg.input(og) test_output = ffmpeg.output(test_input, 'test_' + og) render_(test_output)
    # movie_v.write_videofile('base.mp4', ffmpeg_params=['-c:v', 'h264', '-c:a', 'aac'])
    # movie_v.show()
    movie_v_duration = movie_v.duration
    duration = None
    try:
        duration = movie_v.duration
    except:
        print(colored('Failed to open clip \'' + og + '\' and find its length!', print_color_error))
        return False
    if k_xi(concat_log):
        if verbose: print(colored('Found documentation of what clips to use...', print_color))
        tc_v = []
        tc_a = []
        with open(concat_log, "r") as file:
            for line in file:
                line = line.replace('subclip', '')
                start = float(line.split('t')[0])
                cap = float(line.split('t')[1])
                tmp_v = movie_v.subclip(start, cap)
                tmp_v.fps = movie_v.fps
                tc_v.append(tmp_v)
                tmp_a = movie_a.subclip(start, cap)
                tc_a.append(tmp_a)
                tmp_v.close()
                tmp_a.close()
                del tmp_v
                del tmp_a
                print(colored('Section ' + str(start) + ' to ' + str(cap), 'green'))
        if len(tc_v) < 1:
            print(colored("No packets came through.", print_color_error))
            return False
        processed_v = None
        processed_a = None
        if len(tc_v) > 1:
            if verbose: print(colored('Concatenating snippets...', print_color))
            if verbose: print(colored('Concatenating a list of files from a chunk...', print_color))
            processed_a = concatenate_audioclips(tc_a)
            processed_v = concatenate_videoclips(tc_v)
            print(colored("Concatenating accepted packets.", 'green'))
        else:
            processed_v = tc_v[0]
            processed_a = tc_a[0]
            if verbose: print(colored(
                'Only one packet made it through the vetting process... '
                + 'you should consider lowering your threshold value or changing the search distance value.',
                'blue'))
    else:
        if verbose: print(colored('No documentation found: creating clips and documentation from scratch...', print_color))
        # create averaged audio sections and decide which ones meet threshold
        if verbose: print(colored('Chunking audio...', print_color))
        chunk_length_ms = c_l
        chunk_length_s = chunk_length_ms / 1000
        # maybe use iter_chunks
        chunks_a = make_chunks(movie_a_fc, chunk_length_ms)
        chunks_a_voice = make_chunks(a_voices_fc, chunk_length_ms)
        tc_v = []
        tc_a = []
        list_db_spread = []
        list_db_solo = []
        kc = []
        spread_calc = int((spread / 2))
        # create array of sound levels
        for o in range(0, len(chunks_a) - 1):
            #i=0, l=[], sl=1, sr=1, t_s=0, t_f=1, dud=False
            c = k_chunk(o, chunks_a, spread_calc, spread_calc, o * chunk_length_s, (o + 1) * chunk_length_s, name)
            #print('Chunk created: {0}'.format(c))
            kc.append(c)
        # get list stats for pinpointing threshold
        # k_stats returns [f_spread, f_spread[ls // 2], statistics.average(f_spread), max(f_spread), f_solo, f_solo[ls // 2], statistics.average(f_solo), max(f_solo)]
        rem_solo, thresh_solo, rem_spread, thresh_spread, goldenList, avgP = k_stats(kc)

        if len(rem_solo) < 1:
            print('no values passed')
            return False

        #reference = [{'list_spread': rem_spread,
        #         'mean_spread': mean_f_spread, 'median_spread': median_f_spread, 'max_spread': max_f_spread,
        #         'list_solo': rem_solo,
        #         'mean_solo': mean_f_solo, 'median_solo': median_f_solo, 'max_solo': max_f_solo}]

        # log
        if verbose:
            print(colored('Building snippets...', print_color))

        movie_v_fps = movie_v.fps
        inputs = ''

        concatenated_clips = []
        sourceFile = mpye.VideoFileClip(og)
        if (len(goldenList) == 1):
            if verbose:
                print(colored('One value passed!', print_color))
            sub = sourceFile.subclip(c.t_s, c.t_f)
            sub.write_videofile(base_name + '.mp4')
        #if (len(rem_solo) > 1):
        elif not k_xi(base_name + '.mp4'):
            if verbose:
                print(colored('Multiple values passed!', print_color))

            for i in range(len(goldenList)):
                c = goldenList[i]
                if verbose:
                    print('chunk = {0}'.format(c))
                sub = sourceFile.subclip(c.t_s, c.t_f)
                concatenated_clips.append(sub)
                try:
                    tmp_sub.close()
                except:
                    if verbose:
                        print(colored('Error closing clip during concatenation process!', print_color_error))

            concatenated_clip = mpye.concatenate_videoclips(concatenated_clips)
            concatenated_clip.write_videofile(base_name + '.mp4')


    ret = ffmpeg.input(base_name + '.mp4')
    movie_height = movie_v.h
    desired_height = crop_h
    movie_width = movie_v.w
    desired_width = crop_w
    scale_factor = min((movie_height / desired_height), (movie_width / desired_width))
    if verbose:
        print(colored('Resizing video with a scale factor = {0}\nand dimensions desired_width = {1}\nand desired_height = {2}' \
            .format(scale_factor, desired_width, desired_height), print_color))


    base_v = ret.split()[0].filter('crop', x=1.50 * crop_w * scale_factor, y=0 * crop_h * scale_factor, w=crop_w * scale_factor, h=crop_h * scale_factor)
    output_file = 'fclips\\filtered_and_processed_output_from_' + name
    #output = ffmpeg.output(base_v, , '{0}.mp4'.format(output_file))
    #render_(output)
    base_a = ret.audio \
        .filter("afftdn", nr=6, nt="w", om="o") \
        .filter("afftdn", nr=2, nt="w", om="o") \
        .filter("loudnorm")
    if verbose:
        print(colored('Writing filtered files...', print_color))
    #out_stream = ffmpeg.map_audio(base_v, base_a)
    output = ffmpeg.output(base_v, base_a, '{0}.mp4'.format(output_file))
    render_(output)
    # output_v = ffmpeg.output(base_v, output_file + '_2.mp4')
    # output_a = ffmpeg.output(base_a, output_file + '.mp3')
    # render_(output_v)
    # render_(output_a)
    # subprocess.call('ffmpeg -y -i ' + output_file + '_2.mp4' + ' -i ' + output_file + '.mp3'
    #                + ' -fs 1GB -c:v libx265 -c:a aac -map 0:v:0 -map 1:a:0 ' + output_file + '.mp4')
    # subprocess.call('-c:v libx265 -c:a aac')
    # k_remove(output_file + '_2.mp4')
    clips_to_remove.append(output_file + '.mp4')
    clips_to_remove.append(output_file + '.mp3')
    return output_file + '.mp4'


# make new function that takes the cut times and adds timewarping

# convert a video to an mp4
def to_mp4(name): #-fs 1GB
    subprocess.call('ffmpeg -y -i "{0}" -c:v copy -c:a copy -strict 1 "{1}"' \
        .format(name, name[:-4] + '.mp4'))


# create video timestamps
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


# create video list
def create_video_list(a, ts=False):
    tmp = []
    for name in os.listdir(a): # + '\\' + subfolder):
        fullName = name
        #if len(subfolder) > 0: fullName = subfolder + '\\' + name print(name)
        if os.path.isfile(fullName):
            if verbose:
                print(colored('Found file \"' + name + '\"', print_color))
            name_lower = name.lower()
            name_root = name_lower[:-4]
            name_ext = name_lower[-4:]
            if k_xi('fclips\\filtered_and_processed_output_from_' + name):
                if verbose:
                    print('skipping clip {0}'.format('fclips\\filtered_and_processed_output_from_' + name))
                continue
            if name_ext in ['.mp3', '.wav', '.zip']:
                continue
            if name_ext in ['.m2ts', '.mov']:
                to_mp4(fullName)
                os.rename(fullName, 'not_mp4\\' + name)
                name = name[:-4] + '.mp4'
            if 'subclip' not in name_root and 'output' not in name_root:
                tmp.append(fullName[:-4] + '.mp4')
            if ts:
                create_timestamps(name)
    return tmp


def get_length(filename):
    result = subprocess.Popen(["ffprobe", filename], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return [x for x in result.stdout.readlines() if "Duration" in x]


# sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# with tf.device("/cpu:0"):
# print("current device = " + str(torch.cuda.current_device()))
# with torch.cuda.device(1):
main()
