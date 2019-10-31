from moviepy import *
import moviepy.editor as mpy
filename = 'footage\\GOPR0174_comp.mp4'
tmp_clip = mpy.VideoFileClip(filename)
print(tmp_clip)
file_length = tmp_clip.duration
tmp_clip.reader.close()
tmp_clip.close()
print(tmp_clip)
del tmp_clip
