import ffmpeg

sub = ffmpeg.input('C:\\Users\\kessl\\Desktop\\Code 2019\\kss\\tests\\a.mp4')
tmp_sub = ffmpeg.input('C:\\Users\\kessl\\Desktop\\Code 2019\\kss\\tests\\b.mp4')
ffmpeg.concat(sub['v'], sub['a'], tmp_sub['v'], tmp_sub['a'], v=1, a=1).output('C:\\Users\\kessl\\Desktop\\Code 2019\\kss\\tests\\c.mp4').run()
