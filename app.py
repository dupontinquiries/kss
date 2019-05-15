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

def trim_silent(a, b):
    tmp_name = vid_arr[b]
    print(str(get_length(tmp_name)))
    c = get_chunk_times(tmp_name, 24, 7, 0, get_length(tmp_name))
    print('chunks = ' + str(c))
    return a

def _logged_popen(cmd_line, *args, **kwargs):
    logger.debug('Running command: {}'.format(subprocess.list2cmdline(cmd_line)))
    return subprocess.Popen(cmd_line, *args, **kwargs)

def get_chunk_times(in_filename, silence_threshold, silence_duration, start_time=None, end_time=None):
    input_kwargs = {}
    if start_time is not None:
        input_kwargs['ss'] = start_time
    else:
        start_time = 0.
    if end_time is not None:
        input_kwargs['t'] = end_time - start_time

    p = _logged_popen(
        (ffmpeg
            .input(in_filename, **input_kwargs)
            .filter('silencedetect', n='{}dB'.format(silence_threshold), d=silence_duration)
            .output('-', format='null')
            .compile()
        ) + ['-nostats'],  # FIXME: use .nostats() once it's implemented in ffmpeg-python.
        stderr=subprocess.PIPE
    )
    output = p.communicate()[1].decode('utf-8')
    if p.returncode != 0:
        sys.stderr.write(output)
        sys.exit(1)
    logger.debug(output)
    lines = output.splitlines()

    # Chunks start when silence ends, and chunks end when silence starts.
    chunk_starts = []
    chunk_ends = []
    for line in lines:
        silence_start_match = silence_start_re.search(line)
        silence_end_match = silence_end_re.search(line)
        total_duration_match = total_duration_re.search(line)
        if silence_start_match:
            chunk_ends.append(float(silence_start_match.group('start')))
            if len(chunk_starts) == 0:
                # Started with non-silence.
                chunk_starts.append(start_time or 0.)
        elif silence_end_match:
            chunk_starts.append(float(silence_end_match.group('end')))
        elif total_duration_match:
            hours = int(total_duration_match.group('hours'))
            minutes = int(total_duration_match.group('minutes'))
            seconds = float(total_duration_match.group('seconds'))
            end_time = hours * 3600 + minutes * 60 + seconds

    if len(chunk_starts) == 0:
        # No silence found.
        chunk_starts.append(start_time)

    if len(chunk_starts) > len(chunk_ends):
        # Finished with non-silence.
        chunk_ends.append(end_time or 10000000.)

    print('chunk_starts = ' + str(chunk_starts))

    return list(zip(chunk_starts, chunk_ends))

#start of code

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__file__)
logger.setLevel(logging.INFO)

silence_start_re = re.compile(' silence_start: (?P<start>[0-9]+(\.?[0-9]*))$')
silence_end_re = re.compile(' silence_end: (?P<end>[0-9]+(\.?[0-9]*)) ')
total_duration_re = re.compile(
    'size=[^ ]+ time=(?P<hours>[0-9]{2}):(?P<minutes>[0-9]{2}):(?P<seconds>[0-9\.]{5}) bitrate=')

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
dir = "C:\\Users\\bluuc\\Desktop\\Code 2019\\Eclipse\\Test2\\footage"
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
