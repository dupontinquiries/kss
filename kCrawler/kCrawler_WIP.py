import os
import shutil

from kPath import kPath

def kIn(a):
    b = input(a)
    print(b)
    return b


def compressDir(root, inD, outD, fs, completedConversions=set(), failedConversions=set()):
    """
    root = hard path of parent folder
    inD  = directory of videos to compress; changes with recursion
    outD = output folder to deliver to; constant with recursion
    fs   = file size limit to deliver to ffmpeg processing
    """
    print('inD = {0}'.format(inD))
    print('outD = {0}'.format(outD))
    for name in os.listdir(inD.aPath()):
        print('inD = {0}'.format(inD.aPath()))
        fPath = kPath(inD.append(name))
        #print('{0}, {1}, {2}'.format(root.aPath(), inD.aPath(), outD.aPath()))
        #print('{0}, {1}, {2}'.format(nameRoot, nameExt, fPath.aPath()))
        if fPath.isFile():
            nameRoot = name[:-4]
            nameExt = name[-4:]
            if nameExt in ['.mp3', '.wav', '.zip']:
                continue
            if 'mp4' not in nameExt.lower():
                continue
            if fPath.aPath() in completedConversions:
                continue
            out = outD.append(nameRoot).hitch('_kCrawler.mp4')
            if out.exists():
                continue
            cmd = 'ffmpeg -y -i "{0}" -c:v libx265 -c:a aac "{1}"' \
                .format(fPath, out)
            os.system('{0}'.format(cmd))
            completedConversions.add(fPath.aPath())
        else:
            compressDir(root, inD.append(name), outD, fs, completedConversions, failedConversions, name) #add path changes here as a string and append it to the outD


root = 'M:\\2019\\Recordings 2019\\GoPro\\2019-11-09\\HERO8 BLACK 1'
root = kPath(kIn('Path to the workspace => '))

fs = False
if fs:
    print('\nNext, we will set the amount of compression in terms of file size.\n'
        + 'This method is not very accurate, but lower numbers will '
        + 'increase the amount of compression.\n')
    fs = int(kIn('Enter a file size limit in MB => '))

outD = kPath(root.append('../program_results'))

if not os.path.exists(outD.aPath()):
    os.mkdir(outD.aPath())

os.chdir(root.aPath())

completedConversions = set()
failedConversions = set()

compressDir(root, root, outD, fs)
