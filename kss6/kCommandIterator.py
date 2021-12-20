import argparse

def iterate(p, c, t, i):
    from pathlib import Path
    p = Path(p)
    files = [h for h in p.iterdir() if h.is_file()]
    folders = [h for h in p.iterdir() if h.is_dir()]
    conversions = dict() # add (filename [str], success/failure [bool]) pairs
    for h in files:
        path = p.joinpath(h)
        if str(path.suffix.lower()) != t:
            continue
        a = c.replace("(f)", str(path)).replace("(fr)", str(path.stem)).replace("(fe)", str(path.suffix))
        # print(a)
        # exit()
        import os
        os.system(a)
        # import subprocess as sp
        # sp.call(a)
    if i:
        for h in folders:
            iterate(p.joinpath(h), c, t)
#        conversions = conversions | iterate(p.joinpath(h), c, t) # only works in python 3.9+
#        conversions = {**conversions, **iterate(p.joinpath(h), c, t)}
    
parser = argparse.ArgumentParser()
parser.add_argument("-p", help="path to the workspace")
parser.add_argument("-t", help="file type")
parser.add_argument("-c", help="command to run with (f) for filename and (fr) & (fe) for the file root & extension")
parser.add_argument("-i", help="include to iterate over subdirectories")
args = parser.parse_args()

from os import chdir
chdir(args.p)

conversions = iterate(args.p, str(args.c), args.t, str(args.i).lower() in ["true", "y", "t", "yes"])

# python3 kss6/kCommandIterator.py -p "/Users/thekitchen/Desktop/photos/allie-usb/MyPhotos" -t ".jpg" -c "python3 /Users/thekitchen/Desktop/kss/kss-photo-tools/compress-photo.py -i '/Users/thekitchen/Desktop/photos/allie-usb/MyPhotos/(fr).jpg' -o '/Users/thekitchen/Desktop/photos/allie-usb/output/(fr).jpg' -r '.5:.5'" -i False
