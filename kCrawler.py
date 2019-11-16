import os

def kIn(a):
    b = input(a)
    print(b)
    return b


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

for name in os.listdir(d):
    inp = name
    print(name)
    if '_comp' in inp:
        continue
    if '.mp4' not in inp.lower():
        continue
    if name in completedConversions:
        continue
    try:
        out = name[:-4] + '_comp.mp4'
        cmd = 'ffmpeg -y -i "{0}" -fs {1}MB "{2}"' \
            .format(inp, fs, out)
        #os.system(cmd)
        os.system("start cmd /c {0}" \
            .format(cmd)) 
        completedConversions.add(name)
    except:
        print('Error compressing video: ({0} => {1}).' \
            .format(name, out))
