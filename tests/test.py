import moviepy.editor

vi = moviepy.editor.VideoFileClip("C:/Users/kessl/Desktop/Code 2019/kss/kss/footage/moviepy_subclip_0_60_from_Rocket League 08.07.2017 - 09.38.58.01_comp.mp4")
vo = vi.subclip('00:00:03.0', '00:00:03.5')
vo.write_videofile("C:/Users/kessl/Desktop/Code 2019/kss/kss/footage/b.mp4")
