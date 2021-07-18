import os
import shutil

import subprocess

from kPath import kPath

def kIn(a):
    b = input(a)
    print(b)
    return b

#error in dir M:\2019\Recordings 2019\GoPro\footage\2019-07-10\HERO5 Black 1 because of second layer of photos

def kFileCharacteristic(filename, ret='all'):
    cmd = 'ffprobe "{0}" -show_format ' \
        .format(filename)
    result = subprocess \
        .Popen(cmd, \
        stdout=subprocess.PIPE,stderr=subprocess.STDOUT)
    results = result.communicate()
    #print(results)
    w = None
    if ret == 'all':
        w = results
    else:
        try:
            w = float(str(results).split(ret)[1].split('\\')[0].replace('=', ''))
        except:
            return False
    result.kill()
    return w


def compressDir(cmd, fTypes, root, inD, outD, completedConversions, failedConversions, subF):
    """
    root = hard path of parent folder
    inD  = directory of videos to compress; changes with recursion
    outD = output folder to deliver to; constant with recursion
    fs   = file size limit to deliver to ffmpeg processing
    """
    for name in os.listdir(inD.aPath()):
        fPath = kPath(inD.append(name))
        if fPath.isFile():
            nameRoot = name[:-4]
            nameExt = name[-4:]
            if nameExt.lower() not in fTypes:
                continue
            if fPath.aPath() in completedConversions:
                continue
            out = outD
            if subF != '':
                out = out.append(subF)
            if not out.exists():
                os.mkdir(out.aPath())
            cmd = cmd.replace('(f)', fPath.aPath()).replace('(fr)', nameRoot).replace('(fe)', nameExt)
            print(cmd)
            os.system('{0}'.format(cmd))
            completedConversions.add(fPath.aPath())
        else:
            subFB = subF
            if subF != '':
                subF = subF + '\\'
            subF = subF + name
            compressDir(cmd, root, inD.append(name), outD, completedConversions, failedConversions, subF) 
            subF = ''


root = kPath(kIn('Path to the workspace => '))

cmd = kIn('cmd to run with (f) for filename and (fr), (fe) for root & extension => ')
fTypes = (kIn('file types, separated by spaces, to select for => ')).strip().split(' ')
fTypes = list(filter(lambda x: len(x) > 0, fTypes))

outD = kPath(root.append('../program_results'))

if not outD.exists():
    outD.make()

os.chdir(root.aPath())

completedConversions = set()
failedConversions = set()

compressDir(cmd, fTypes, root, root, outD, set(), set(), '')
