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

from numba import vectorize

t_stamps = {}

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
    pipe = subprocess.Popen(command, stdout=subprocess.PIPE, bufsize=10**8)
    return pipe

def write_audio_file(file_path,audio_array):
    '''
    Need to do this just for testing
    want to be able to create some bizzarre test files for croma print
    '''
    pipe = subprocess.Popen([ FFMPEG_BIN,
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
                    stdin=subprocess.PIPE,stdout=subprocess.PIPE, stderr=sys.stdin)

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

def chunk_file(file_name, max_chunk_size = 10, file_suffix = "default", extention= ".aac"):
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
        if str("k_chunk_n=") not in name: #and name.endswith(".h264")
            input = ffmpeg.input(name)
            name_split = name.split(".")[0] #now can only have one period in whole name
            output = ffmpeg.output(input, name_split + ".h264")
            ffmpeg.run(output)
            os.rename(name, 'not_h264\\' + name)
            #input = ffmpeg.input(name_split + ".h264")
            chunk_file(name_split + ".h264", max_chunk_size)

def read_in_chunks(file_object, chunk_size=1024):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    while True:
        data = file_object.read(chunk_size)
        if not data:
            break
        yield data

def read_in_ffmpeg_chunks(filename, max_chunk_size):
    """Lazy function (generator) to read a file piece by piece.
    Default chunk size: 1k."""
    max_chunk_size *= 60
    t_s = 0
    t_f = max_chunk_size
    while True:
        name = str('moviepy_subclip_' + str(t_s) + '_from_' + str(filename))
        sub = moviepy.video.io.ffmpeg_tools.ffmpeg_extract_subclip(str(filename), t_s, t_f,
                                                                   targetname=name)
        t_s += max_chunk_size
        t_f += max_chunk_size
        ret = ffmpeg.input(name)
        if not ret:
            break
        yield [ret, name]



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

def distr(filename, mod, c_l, spread, thresh_mod = 0.9, crop_w = 1080, crop_h = 1350, max_chunk_size = 5):
    smaller_clips = []
    for piece in read_in_ffmpeg_chunks(filename, max_chunk_size):
        print('piece: ' + str(piece))
        result = process_audio_loudness_over_time(piece[0], piece[1], mod, c_l, spread, thresh_mod, crop_w, crop_h)
        smaller_clips.append(result)
    total = None
    #for vid in smaller_clips:z
    #    if total is None:
    #        total = vid
    #    else:
    #        total += vid
    #    print('add together')
    #render out
    #only renders out first chunk
    total = ffmpeg.merge_outputs(smaller_clips)
    output = ffmpeg.output(total, 'completed_file_of_' + filename)
    ffmpeg.run(output, overwrite_output=True)
    return ffmpeg.input('completed_file_of_' + filename)

#merge_outputs = combine clips; overwrite_output = overwrite files /save lines of code
def process_audio_loudness_over_time(i, name, mod, c_l, spread, thresh_mod, crop_w, crop_h):
    #clean up files space
    if os.path.exists("final\\processed_output_from_" + name + ".h264"):
       os.remove("final\\processed_output_from_" + name + ".h264")
    if os.path.exists("final\\filtered_and_processed_output_from_" + name + ".h264"):
       os.remove("final\\filtered_and_processed_output_from_" + name + ".h264")
    if os.path.exists("tmp_a_from_" + name + ".aac"):
       os.remove("tmp_a_from_" + name + ".aac")
    if os.path.exists("tmp_voice_opt_from_" + name + ".aac"):
       os.remove("tmp_voice_opt_from_" + name + ".aac")
    #remove .h264 to use other filetypes like .aac
    name = str(name.replace(".h264", ""))
    input = i
    a = input['a']
    #clean up audio so program takes loudness of voice into account moreso than other sounds
    a_voice = a.filter('highpass', 300).filter("lowpass", 10000).filter("loudnorm")
    #clean up audio of final video
    a = a.filter('highpass', 50).filter("lowpass", 16000).filter("loudnorm")
    v = input['v']
    #new try; using only what I have already imported to filter through


    #print('length of [\'v\'] = ' + len(v))
    print('sub[0] = ' + str(i))
    print('sub[1] = ' + name)


    #end of new try
    #deprecated; used to get subclips in the processing part
    movie = VideoFileClip(name + ".h264")
    #export clip audio
    output = ffmpeg.output(a, "tmp_a_from_" + name + ".aac")
    ffmpeg.run(output, overwrite_output=True)
    #export voice_optimized audio
    output = ffmpeg.output(a, "tmp_voice_opt_from_" + name + ".aac")
    ffmpeg.run(output, overwrite_output=True)
    #import the new audio
    a = AudioSegment.from_mp3("tmp_a_from_" + name + ".aac")
    k_remove("tmp_a_from_" + name + ".aac")
    a_voices = AudioSegment.from_mp3("tmp_voice_opt_from_" + name + ".aac")
    k_remove("tmp_voice_opt_from_" + name + ".aac")
    #remove file that was rendered
    k_remove("tmp_voice_opt_from_" + name + ".aac")
    #create averaged audio sections and decide which ones meet threshold
    chunk_length_ms = c_l
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
    average_db = statistics.mean(list_of_db)
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
    processed.write_videofile("final\\processed_output_from_" + name + ".h264")
    ret = ffmpeg.input("final\\processed_output_from_" + name + ".h264")
    movie_width = movie.w
    movie_height = movie.h
    desired_height = crop_h
    movie_width = movie.w
    desired_width = crop_w
    scale_factor = max((movie_height / desired_height), (movie_width / desired_width))
    print("scale factor")
    print("scale factor = " + str(scale_factor))
    print("scale factor")
    base_a = ret['a'].filter("loudnorm").filter("acompressor")  # .filter("dynaudnorm")
    ffmpeg.run(ffmpeg.output(base_a, "final\\processed_audio_from_" + name + ".aac"), overwrite_output=True)
    new_audio = (ffmpeg.input("final\\processed_audio_from_" + name + ".aac"))
    base_v = ret['v'].filter("atadenoise").filter('crop', x=0*crop_w*scale_factor, y=0, w=crop_w*scale_factor, h=crop_h*scale_factor)
    output = ffmpeg.output(base_v, new_audio, "final\\filtered_and_processed_output_from_" + name + ".h264")
    ffmpeg.run(output, overwrite_output=True)
    ret = ffmpeg.input("final\\filtered_and_processed_output_from_" + name + ".h264")
    #if os.path.exists("final\\processed_output_from_" + i + ".h264"): os.remove("final\\processed_output_from_" + i + ".h264") if os.path.exists("final\\filtered_and_processed_output_from_" + i + ".h264"): os.remove("final\\filtered_and_processed_output_from_" + i + ".h264")
    return ret

def process_audio_loudness_over_time_old(i, mod, c_l, spread, thresh_mod, crop_w, crop_h):
    #if os.path.exists("final\\processed_output_from_" + i + ".h264"):
    #    os.remove("final\\processed_output_from_" + i + ".h264")
    #if os.path.exists("final\\filtered_and_processed_output_from_" + i + ".h264"):
    #    os.remove("final\\filtered_and_processed_output_from_" + i + ".h264")
    i = str(i.replace(".h264", ""))
    if not os.path.exists("tmp_a_from_" + i + ".aac"):
        input = ffmpeg.input(i + ".h264")
        a = input['a']
        a_voice = a.filter('highpass', 300).filter("lowpass", 10000).filter("loudnorm")
        a = a.filter('highpass', 400).filter("lowpass", 15000).filter("loudnorm")
        v = input['v']
        movie = VideoFileClip(i + ".h264")
        output = ffmpeg.output(a, "tmp_a_from_" + i + ".aac")
        ffmpeg.run(output, overwrite_output=True)
    a = AudioSegment.from_mp3("tmp_a_from_" + i + ".aac")
    k_remove("tmp_a_from_" + i + ".aac")
    if not os.path.exists("tmp_voice_opt_from_" + i + ".aac"):
        output = ffmpeg.output(a, "tmp_voice_opt_from_" + i + ".aac")
        ffmpeg.run(output, overwrite_output=True)
    #import the new audio
    a_voices = AudioSegment.from_mp3("tmp_voice_opt_from_" + i + ".aac")
    k_remove("tmp_voice_opt_from_" + i + ".aac")
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
    processed.write_videofile("final\\processed_output_from_" + i + ".h264")
    ret = ffmpeg.input("final\\processed_output_from_" + i + ".h264")
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
    ffmpeg.run(ffmpeg.output(base_a, "final\\processed_audio_from_" + i + ".aac"), overwrite_output=True)
    new_audio = (ffmpeg.input("final\\processed_audio_from_" + i + ".aac"))
    base_v = ret['v'].filter("atadenoise").filter('crop', x=0*crop_w*scale_factor, y=0, w=crop_w*scale_factor, h=crop_h*scale_factor)
    output = ffmpeg.output(base_v, new_audio, "final\\filtered_and_processed_output_from_" + i + ".h264")
    ffmpeg.run(output, overwrite_output=True)
    ret = ffmpeg.input("final\\filtered_and_processed_output_from_" + i + ".h264")
    #if os.path.exists("final\\processed_output_from_" + i + ".h264"): os.remove("final\\processed_output_from_" + i + ".h264") if os.path.exists("final\\filtered_and_processed_output_from_" + i + ".h264"): os.remove("final\\filtered_and_processed_output_from_" + i + ".h264")
    return ret



# make new function that takes the cut times and adds timewarping

# make new function that takes a song and uses the song to determine threshold at the time
# and the cut speed is determined by the time between closest to min and closest to max point (distance) in array
def to_h264(name):
    i = ffmpeg.input(name)
    o = ffmpeg.output(i, 'file--' + name + '--.h264')
    ffmpeg.run(o, overwrite_output=True)
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
            to_h264(name)
            os.rename(name, 'notmp4\\' + name)
    for name in os.listdir(a):
        if name.endswith(".h264"):
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
    i = str(i.replace(".h264", ""))
    a = input['a']
    a_voice = a.filter('highpass', 300).filter("lowpass", 10000).filter("loudnorm")
    a = a.filter('highpass', 400).filter("lowpass", 15000).filter("loudnorm")
    v = input['v']
    output = ffmpeg.output(a, "tmp_a_from_" + i + ".wav")
    ffmpeg.run(output, overwrite_output=True)
    output = ffmpeg.output(a, "tmp_voice_opt_from_" + i + ".wav")
    ffmpeg.run(output, overwrite_output=True)
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
    # output = ffmpeg.output(base['v'], base['a'], "processed_output_from_" + i + ".h264")
    # ffmpeg.run(output, overwrite_output=True)
    processed = concatenate(tc_v)
    print("concat: " + str(processed))
    processed.write_videofile("final\\processed_output_from_" + i + ".h264")
    ret = ffmpeg.input("final\\processed_output_from_" + i + ".h264")
    movie_width = movie.width
    movie_height = movie.height
    desired_height = 1350
    scale_factor = movie_height/desired_height
    base_a = ret['a'].filter("loudnorm").filter("acompressor")  # .filter("dynaudnorm")
    base_v = ret['v'].filter("atadenoise").filter('scale', w=(str(movie_width*scale_factor)),
                                                  h=(str(movie_height*scale_factor))).filter('crop', w='1080', h='1350')
    ffmpeg.run(base_v, base_a, "final\\filtered_and_processed_output_from_" + i + ".h264", overwrite_output=True)
    ret = ffmpeg.input("final\\filtered_and_processed_output_from_" + i + ".h264")
    #if os.path.exists("processed_output_from_" + i + ".h264"):
        #os.remove("processed_output_from_" + i + ".h264")
    return ret


print(str(ffmpeg) + " is running")

dir = dirname(abspath(__file__)) + "\\footage"
print("root: " + str(dir))
os.chdir(dir)
vid_arr = create_video_list(dir, False)
#vid_arr.sort(key=lambda x: os.path.getmtime(x))
print("list: " + str(vid_arr) + "")
#base = ffmpeg.input(vid_arr[0])
base = distr(vid_arr[0], 0.7, 300, 5, 1.35, 1920, 1080, 3)
base_v = base['v']
base_a = base['a']
if(len(vid_arr) > 1):
    for w in range(1, len(vid_arr) - 1):
        # concat = trim_silent(ffmpeg.input(vid_arr[w+1]), w)
        to_concat = distr(vid_arr[w], 0.7, 300, 5, 1.35, 1920, 1080, 3) #1.4, 15000, 5 is pretty good with about 50% retention and near full comprehensibility
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
# base_a = ffmpeg.input("PowerStone.aac")['a']
out = ffmpeg.output(base_v, base_a, "final\\output_from_" + "all_clips" + ".h264")
ffmpeg.run(out, overwrite_output=True)
print("done")
# to fix len issue? https://stackoverflow.com/questions/19182188/how-to-find-the-length-of-a-filter-object-in-python
# sort video files by date filmed
# https://stackoverflow.com/questions/168409/how-do-you-get-a-directory-listing-sorted-by-creation-date-in-python