import subprocess

subprocess.run('ffmpeg -h')


###


subClip = mpye.VideoFileClip(filename)
try:
    subClip.close()
except:
    if verbose:
        print('failed to close input = {0}'.format(filename))
subClip = subclip(t_s, min(t_f, file_length - t_s))
subClip.write_videofile(name)
