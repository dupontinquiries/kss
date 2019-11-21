import os

def kIn(a):
    b = input(a)
    print(b)
    return b


def compressDir(dir):
    for name in os.listdir(dir):
        inp = name
        name_ext = inp[-4:]
        if os.path.isfile(name):
            if name_ext in ['.mp3', '.wav', '.zip']:
                continue
            if 'mp4' not in inp.lower():
                continue
            if name in completedConversions:
                continue
            try:
                out = name[:-4] + '_kCrawler.mp4'
                cmd = 'ffmpeg -y -i "{0}" -fs {1}MB "{2}"' \
                    .format(inp, fs, p + '\\' + out)
                #os.system(cmd)
                os.system('{0}' \
                    .format(cmd))
                completedConversions.add(name)
            except:
                print('Error compressing video: ({0} => {1}).' \
                    .format(name, out))
        else:
            compressDir(name)


d = 'M:\\2019\\Recordings 2019\\GoPro\\2019-11-09\\HERO8 BLACK 1'
d = kIn('Path to the workspace => ')
print('\nNext, we will set the amount of compression in terms of file size.\n'
    + 'This method is not very accurate, but lower numbers will '
    + 'increase the amount of compression.\n')
fs = int(kIn('Enter a file size limit in MB => '))
print(d)
assert os.path.exists(d), \
    'Path "{0}" not found.'.format()
os.chdir(d)

completedConversions = set()
failedConversions = set()

p = 'program_results'
if not os.path.exists(p):
    os.mkdir(p)

compressDir(d)


if not os.path.exists('../' + p):
    os.mkdir('../' + p)

shutil.move(p, '../' + p)
