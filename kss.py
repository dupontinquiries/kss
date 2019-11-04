from __future__ import unicode_literals
import statistics
import cv2
import ffmpeg
from moviepy.video.io.ffmpeg_tools import *
import moviepy
import moviepy.editor as mpye
from os.path import dirname, abspath
import subprocess
from pydub import AudioSegment
from pydub.utils import make_chunks
#from moviepy.editor import concatenate_videoclips, concatenate_audioclips, VideoFileClip, AudioFileClip
#from moviepy.editor import *
# from moviepy import write_videofile
# from math import *
# import numpy
from termcolor import colored
# import tensorflow-gpu as tf
import random
import os
import sys
from k_chunk import k_chunk

# import torch as torch

# from numba import vectorize

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
DEFAULT_THRESHOLD = .3
DEFAULT_PERIOD = 750
DEFAULT_REACH_ITER = 2
DEFAULT_REACH_THRESH = .3
DEFAULT_WIDTH = 1920  # 2560
DEFAULT_HEIGHT = 1080  # 1440
DEFAULT_MAX_CHUNK_SIZE = 20 #1.2, 3.2, 10.2
DEFAULT_TREATMENT = list(['voice', 'music'])[0]
verbose = True
cleanup = True
dir = ''


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
        print(colored('The directory \"' + dir + '\" was not found!', 'red'))
        sys.exit(0)
    os.chdir(dir)
    print('dir = {0}'.format(os.getcwd()))
    if verbose: print(colored('root: ' + str(dir), 'blue'))
    if verbose: print(colored('Finding all video files...', 'blue'))
    vid_arr = create_video_list(dir,
                                False)  # dir, create time stamps and reorder based on time blocks, time block duration
    if len(vid_arr) < 1:
        print(colored('No video files were found in \"' + dir + '\"!', 'red'))
        sys.exit(0)
    if verbose: print(colored('Processing files...', 'blue'))
    # gather clips for main file
    maxi = max(1, len(vid_arr) - 1)
    for w in range(0, max(1, len(vid_arr) - 1)):
        # concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
        process = distr(vid_arr[w], THRESHOLD, PERIOD, REACH_ITER, REACH_ITER, WIDTH, HEIGHT, MAX_CHUNK_SIZE)
        if process != False:
            final_cuts.append(process)
            if verbose:
                print(colored('Adding file \"' + str(process) + '\" to the final video...', 'blue'))
        else:
            print(colored('An error was encoutered while adding file #' + str(w) + ' to the final video!', 'red'))
    if verbose: print(colored('Your final video is almost ready...', 'blue'))
    main = None
    if len(final_cuts) > 1:
        main = concatenate_videoclips(final_cuts)
    if len(final_cuts) == 1:
        main = final_cuts[0]
    else:
        print(colored('No clips rendered!', 'red'))
        sys.exit(0)
    if verbose: print(colored('Your final video has a length of ' + str(main.duration) + '!', 'blue'))
    main.write_videofile('final\\final_cut_with_props_' + str(THRESHOLD) + '__' + str(PERIOD) + '__' + str(REACH_ITER)
                         + '__' + str(WIDTH) + '__' + str(HEIGHT) + '__' + str(MAX_CHUNK_SIZE) + '.mp4')
    # clean up all clips
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


def getLength(filename):
    cmd = 'ffprobe "{0}" -show_format ' \
        .format(filename)
    result = subprocess \
        .Popen(cmd, \
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    results = result.communicate()
    #print(results)
    file_length = float(str(results).split('nduration')[1].split('\\')[0].replace('=', ''))
    return file_length


def read_in_ffmpeg_chunks(filename, max_chunk_size):
    file_length = 0
    if verbose:
        print(colored('Finding length...', 'blue'))
    file_length = getLength(filename)
    print('d = {0}'.format(file_length))
    max_chunk_size *= 60  # convert to seconds
    t_s = 0
    t_f = min(file_length, max_chunk_size)
    # itereate through the file and make chunks
    while file_length - t_s > 0:
        delta = t_f - t_s
        print("t_s = " + str(t_s) + "; " + "t_f = " + str(t_f) + "; " + "d = " + str(delta) + "; ")
        if file_length - t_f <= 0:
            yield False
        name = 'moviepy_subclip_{0}_{1}_from_{2}'\
            .format(t_s, t_f, filename)
        # generates a subclip to avoid memory caps
        # saves time on reruns to skip if the file already exists
        if not k_xi(name):
            # export subclip
            cmd = 'ffmpeg -y -i "{0}" -ss {1} -t {2} -vcodec h264 -acodec aac "{3}"'\
                .format(filename, t_s, min(delta, file_length - t_s), name)
            print('[cmd] ~ {0}'.format(cmd))
            subprocess.call(cmd)
            clips_to_remove.append(name)
        else:
            print('skipping chunk-rendering of clip \"' + name + '\"')
        t_s += delta
        t_f += delta
        # ret = VideoFileClip(name)
        # print('nav = ' + str(ret))
        ret = ffmpeg.input(name)
        if ret == None:
            yield False
        yield [ret, name]


def k_remove(a):
    if k_xi(a):
        os.remove(a)


# concatenating video files
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
            b = ffmpeg.concat(b['v'], b['a'], c['v'], c['a'], v=1, a=1)
        return b


def k_path(p):
    return str(os.path.abspath(str(p)))


# concatenating audio files
def k_map(a, name):
    if verbose: print(colored('Rendering output \'' + name + '\' with the k_map function...', 'blue'))
    b = a[0]
    if len(a) == 0:
        return None
    if len(a) == 1:
        inp = ffmpeg.input(b)
        #outp = ffmpeg.output(inp, name)
        inp = VideoFileClip(name)
        return inp
    else:
        if verbose: print(colored('Concatenating list of files...', 'blue'))
        guide_file = 'chunks\\files_to_concat.txt'
        k_remove(guide_file)
        with open(guide_file, "w") as file:
            inputs = ''
            print('a = ' + str(a))
            for x in range(0, len(a) - 1):
                fn = a[x]
                fn = fn.replace('chunks\\', '')
                inputs += 'file \'' + str(fn) + '\'\n'
            if verbose: print(colored('input list: \n' + inputs, 'blue'))
            file.write(inputs)
        cmd = ('ffmpeg -y -f concat -safe 0 -i \"' + k_path(guide_file).replace('\\', '/')
               + '\" -fs 1GB -c:v copy -c:a aac \"' + k_path(name).replace('\\', '/') + '\"')
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
    v_t = concatenate_audioclips(mpys_v)
    a_t = concatenate_videoclips(mpys_a)
    v_t.set_audio(a_t)
    v_t.write_videofile(output_name)

def distr(filename, mod, c_l, spread, thresh_mod, crop_w, crop_h, max_chunk_size):
    base_name = filename[:-4]
    # compress any large files
    smaller_clips = []
    if verbose: print(colored('Verifying clip \"' + filename + '\"', 'blue'))
    if "completed" in filename or "output_from_all" in filename or "sublcip" in filename or "moviepy" in filename:
        return False
    tmp_clip = False
    # get duration
    if verbose: print(colored('Finding length...', 'blue'))
    tmp_clip = mpye.VideoFileClip(filename)
    l = tmp_clip.duration
    del tmp_clip
    if not l > 0:
        console_a = ''
        if not k_xi(filename):
            console_a = ' not'
        print(colored('An error was encountered while opening \"'
                      + filename + '\"!  The file seems to' + console_a + ' exist.', 'red'))
        sys.exit(0)
    if verbose: print(colored('Chunking clip...', 'blue'))
    for piece in read_in_ffmpeg_chunks(filename, max_chunk_size):
        if piece is not False:
            input = piece[0]
            fn = piece[1]
            if file_size(fn) >= (10 ** 9):
                if verbose: print("file " + fn + " is large (" + str(file_size(fn))
                                  + ").  (Future Capability) Keeping the chunked clips as \"cc\"")
            if verbose: print('Opening chunk \"' + fn + '\"')
            result = process_audio_loudness_over_time(input, fn, mod, c_l, spread, thresh_mod, crop_w, crop_h)
            if result is not False:
                smaller_clips.append(result)
                if verbose: print(colored('Adding clip \"' + fn + '\" to the list...', 'blue'))
            else:
                print('No data appended for chunk \"' + fn + '\"')
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
    n = 0
    while fn(tmp[n]) < tr and n < len(l) - 1:
        n += 1
    return tmp[n:]


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
    w = k_chunk(0, None, 0, 0, 0, 0, True).DEFAULT_FLOOR
    thresholds = \
    (DEFAULT_THRESHOLD * ((averages[0] * .3) + (medians[0].v * .4) + (maxes[0].v * .3)), \
    DEFAULT_REACH_THRESH * ((averages[1] * .3) + (medians[1].sv * .4) + (maxes[1].sv * .3)))
    end = datetime.now()
    if verbose: print('time to generate thresholds: {0}'.format(end - start))
    if verbose: print('thresholds = {0}'.format(thresholds))

    start = datetime.now()
    rem_solo = k_splval(f_solo, thresholds[0], get_ts)
    rem_spread = k_splval(f_spread, thresholds[1], get_ts)
    end = datetime.now()
    if verbose: print('time to generate trimmed lists: {0}'.format(end - start))
    if verbose: print('len(solo) = {0}\nlen(rem_solo) = {1}\npercentage = {2:.3f}' \
        .format(len(f_solo), len(rem_solo), len(rem_solo) / len(f_solo)))
    if verbose: print('len(spread) = {0}\nlen(rem_spread) = {1}\npercentage = {2:.3f}' \
        .format(len(f_spread), len(rem_spread), len(rem_spread) / len(f_spread)))

    k_quick_sort(rem_solo, 0, len(rem_solo) - 1, get_ts)
    k_quick_sort(rem_spread, 0, len(rem_spread) - 1, get_ts)
    #if verbose: print('ordered solo list:\n{0}\nordered spread list:\n{1}' \
    #    .format(rem_solo, rem_spread))

    #exit()
    # TODO:
    # make dictionary of values anmd pass nack to parent function
    # add recursive tuning loop that can adjust threshold value by abs(.1 * floor) value

    return (rem_solo, thresholds[0], rem_spread, thresholds[1])


# merge_outputs = combine clips; overwrite_output = overwrite files /save lines of code
def process_audio_loudness_over_time(input, name, mod_solo, c_l, spread, mod_multi, crop_w, crop_h):
    # create log for future renderings
    concat_log = 'chunks\\concat_log_args_' + str(name) + '__' + str(mod_solo) + '__' + str(c_l) + '__' + str(spread) \
                 + '__' + str(mod_multi) + '__' + str(crop_w) + '__' + str(crop_h) + '.txt'
    # get root of file name
    og = name
    name = name[:-4]
    # audio
    name_audio = 'chunks\\tmp_a_from_' + name + '.wav'
    name_audio_voice = 'chunks\\tmp_voice_opt_from_' + name + '.wav'
    if verbose: print(colored('Preparing audio for video...', 'blue'))
    # video clip audio
    #set values for treatment type
    highpass = 80
    lowpass = 1200
    if DEFAULT_TREATMENT == 'music':
        highpass = 0
        lowpass = 20000
    if not k_xi(name_audio):
        a_name_audio = input.audio \
            .filter("afftdn", nr=6, nt="w", om="o") \
            .filter("afftdn", nr=2, nt="w", om="o") \
            .filter("loudnorm")
        # clean up audio so program takes loudness of voice into account moreso than other sounds
        # clean up audio of final video
        if verbose: print(colored('Preparing tailored audio...', 'blue'))
        # export clip audio
        output = ffmpeg.output(a_name_audio, name_audio)
        if verbose: print(colored('Writing tailored audio...', 'blue'))
        render_(output)
    if verbose: print(colored('Importing tailored audio from \"' + name_audio + '\"...', 'blue'))
    a = ffmpeg.input(name_audio)
    # voice_opt
    if not k_xi(name_audio_voice):
        if verbose: print(colored('Preparing audio for analysis...', 'blue'))
        a_voice = a.audio \
            .filter("afftdn", nr=6, nt="w", om="o") \
            .filter('highpass', highpass) \
            .filter("lowpass", lowpass) \
            .filter("afftdn", nr=2, nt="w", om="o") \
            .filter("loudnorm")
        # export voice_optimized audio
        output = ffmpeg.output(a_voice, name_audio_voice)
        render_(output)
    if verbose: print(colored('Importing optimized audio from \"' + name_audio_voice + '\"...', 'blue'))
    a_voice = ffmpeg.input(name_audio_voice)
    # import the new voice_opt audio
    if verbose: print(colored('Establishing audio files...', 'blue'))
    movie_a_fc = AudioSegment.from_wav(name_audio)
    a_voices_fc = AudioSegment.from_wav(name_audio_voice)
    if verbose: print(colored('Name of audio file is \"' + name_audio + '\"', 'blue'))


    movie_a = mpye.AudioFileClip(name_audio)
    try:
        movie_a.close()
        movie_a.reader.close()
    except:
        if verbose: print(colored('Error closing reader of optimized audio from \"' + name_audio_voice + '\"...', 'red'))
    #print(movie_a)
    #exit()

    movie_a_length = movie_a.duration
    a_voices = mpye.AudioFileClip(name_audio_voice)
    try:
        a_voices.close()
        a_voices.reader.close()
    except:
        if verbose: print(colored('Error closing reader of optimized audio from \"' + name_audio_voice + '\"...', 'red'))
    # add them to delete list
    clips_to_remove.append(name_audio)
    clips_to_remove.append(name_audio_voice)
    # get subclips in the processing part
    if verbose: print(colored('Opening clip \'' + og + '\'...', 'blue'))
    movie_v = mpye.VideoFileClip(og)
    movie_v.close()
    # test_input = ffmpeg.input(og) test_output = ffmpeg.output(test_input, 'test_' + og) render_(test_output)
    # movie_v.write_videofile('base.mp4', ffmpeg_params=['-c:v', 'h264', '-c:a', 'aac'])
    # movie_v.show()
    movie_v_duration = movie_v.duration
    duration = None
    try:
        duration = movie_v.duration
    except:
        print(colored('Failed to open clip \'' + og + '\' and find its length!', 'red'))
        return False
    if k_xi(concat_log):
        if verbose: print(colored('Found documentation of what clips to use...', 'blue'))
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
            print(colored("No packets came through.", 'red'))
            return False
        processed_v = None
        processed_a = None
        if len(tc_v) > 1:
            if verbose: print(colored('Concatenating snippets...', 'blue'))
            if verbose: print(colored('Concatenating a list of files from a chunk...', 'blue'))
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
        if verbose: print(colored('No documentation found: creating clips and documentation from scratch...', 'blue'))
        # create averaged audio sections and decide which ones meet threshold
        if verbose: print(colored('Chunking audio...', 'blue'))
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
            c = k_chunk(o, chunks_a, spread_calc // 2, spread_calc, o * chunk_length_s, (o + 1) * chunk_length_s)
            #print('Chunk created: {0}'.format(c))
            kc.append(c)
        # get list stats for pinpointing threshold
        # k_stats returns [f_spread, f_spread[ls // 2], statistics.average(f_spread), max(f_spread), f_solo, f_solo[ls // 2], statistics.average(f_solo), max(f_solo)]
        rem_solo, thresh_solo, rem_spread, thresh_spread = k_stats(kc)

        exit()

        #reference = [{'list_spread': rem_spread,
        #         'mean_spread': mean_f_spread, 'median_spread': median_f_spread, 'max_spread': max_f_spread,
        #         'list_solo': rem_solo,
        #         'mean_solo': mean_f_solo, 'median_solo': median_f_solo, 'max_solo': max_f_solo}]


        if False:
            l_db_spread = kstats[0]
            average_db_spread = kstats[1]
            median_db_spread = kstats[2]
            max_db_spread = kstats[3]
            # solo
            l_db_solo = kstats[4]
            average_db_solo = kstats[5]
            median_db_solo = kstats[6]
            max_db_solo = kstats[7]
            # get thresholds
            target_db = ((.5 * median_db_solo) + (.4 * average_db_solo) + (.1 * max_db_solo))
            thresh_solo = mod_solo * target_db
            target_db = (.7 * average_db_spread) + (.2 * average_db_spread) + (.1 * max_db_spread)
            thresh_spread = mod_multi * target_db
            # get cutoff values
            cutoffs = k_binary_index(l_db_spread, l_db_solo, thresh_spread, thresh_solo)
            i_cut_spread = cutoffs[0]
            l_a_spread = sorted(l_db_spread[i_cut_spread:], key=lambda k_chunk: k_chunk.i)
            i_cut_solo = cutoffs[1]
            l_a_solo = sorted(l_db_solo[i_cut_solo:], key=lambda k_chunk: k_chunk.i)
            fl = []
            if i_cut_solo < i_cut_spread:
                # proceed with spread list
                l1 = l_a_spread
                l2 = l_a_solo
            else:
                # proceed with solo list
                l1 = l_a_solo
                l2 = l_a_spread
            for o in range(0, len(l1) - 1):
                index = k_l_compare(l2, l1[o])
                if index != -1:
                    fl.append(l1[o])
        # log
        le = len(fl)
        if (le == 0):
            print('no values passed')
            return False
        if verbose: print(colored('Building snippets...', 'blue'))
        movie_v_fps = movie_v.fps
        inputs = ''
        sub = i.trim(start_frame=movie_v_fps * c.t_s, end_frame=movie_v_fps * c.t_f)
        if (le == 1):
            print('one value passed')
        if (le > 1):
            print('multiple values passed')
            for c in fl[1:]:
                tmp_sub = i.trim(start_frame=movie_v_fps * c.t_s, end_frame=movie_v_fps * c.t_f)
                sub = ffmpeg.concat(sub['v'], sub['a'], tmp_sub['v'], tmp_sub['a'], v=1, a=1)
        if verbose: print(
            colored('thresh_solo = ' + str(thresh_solo) + '\nthresh_multi = ' + str(thresh_spread), 'blue'))
        fr = sub
        # p_t = k_round(x / (len(chunks_a) - 1), 5)
        # avg_p = int(statistics.mean(p_arr))
        # periods = ''
        # for o in range(0, avg_p):
        #    periods += '.'
        # print(colored('Accepted ' + str(movie_v_duration * p_t) + 's or ' + str(p_t * 100) + '% of the clip'
        #                + '\nwith an average length of ' + str(avg_p * chunk_length_s) + 's:'
        #                + '\n' + periods
        #                              , 'blue'))
        print('tecca = ' + str(fr))
    base_name = 'chunks\\processed_output_from_' + name
    ret = fr  # ffmpeg.input(base_name + '.mp4')
    movie_height = movie_v.h
    desired_height = crop_h
    movie_width = movie_v.w
    desired_width = crop_w
    scale_factor = min((movie_height / desired_height), (movie_width / desired_width))
    if verbose: print(colored('Resizing video with a scale factor of ' + str(scale_factor) + ', and dimentions w: '
                              + str(desired_width) + ' and h: ' + str(desired_height) + '...', 'blue'))
    base_v = ret.video.filter('crop', x=1.50 * crop_w * scale_factor, y=0 * crop_h * scale_factor,
                              w=crop_w * scale_factor, h=crop_h * scale_factor)
    base_a = ret.audio.filter("loudnorm").filter("afftdn", nr=8, nt="w", om="o").filter("equalizer", f=7000,
                                                                                        width_type="o", width=5,
                                                                                        g=1).filter("equalizer", f=200,
                                                                                                    width_type="o",
                                                                                                    width=2,
                                                                                                    g=-1)
    output_file = 'chunks\\filtered_and_processed_output_from_' + name
    if verbose: print(colored('Writing filtered files...', 'blue'))
    output = ffmpeg.output(base_v, base_a, output_file + '.mp4')
    # output_v = ffmpeg.output(base_v, output_file + '_2.mp4')
    # output_a = ffmpeg.output(base_a, output_file + '.wav')
    # render_(output_v)
    # render_(output_a)
    # subprocess.call('ffmpeg -y -i ' + output_file + '_2.mp4' + ' -i ' + output_file + '.wav'
    #                + ' -fs 1GB -c:v copy -c:a aac -map 0:v:0 -map 1:a:0 ' + output_file + '.mp4')
    # subprocess.call('-c:v copy -c:a aac')
    # k_remove(output_file + '_2.mp4')
    clips_to_remove.append(output_file + '.mp4')
    clips_to_remove.append(output_file + '.wav')
    return output_file + '.mp4'


# make new function that takes the cut times and adds timewarping

# convert a video to an mp4
def to_mp4(name):
    subprocess.call('ffmpeg -y -i ' + name + ' -fs 1GB -c:v copy -c:a aac ' + name[-4:] + '.mp4')


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
    for name in os.listdir(a):
        if (os.path.isfile(name)):
            if verbose: print(colored('Found file \"' + name + '\"', 'blue'))
            name_lower = name.lower()
            name_root = name_lower[:-4]
            name_ext = name_lower[-4:]
            if name_ext in ['.m2ts', '.mov']:
                to_mp4(name)
                os.rename(name, 'not_mp4\\' + name)
                name = name[:-4] + '.mp4'
            if 'subclip' not in name_root and 'output' not in name_root:
                tmp.append(name)
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
