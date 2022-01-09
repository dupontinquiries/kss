import argparse
import os
from pathlib import Path
from os import chdir

def iterate_threaded_helper(p, h, c, t):
    import os
    from pathlib import Path
    path = p.joinpath(h)
    # if t != "*" and str(path.suffix.lower()) != t:
    #     return
    if t == "*" or str(path.suffix.lower()) == t:
        a = c.replace("(f)", str(path)).replace("(fr)", str(path.stem)).replace("(fe)", str(path.suffix))
        
        os.system(a)

def iterate_threaded(nthreads, p, c, t, i, addToPath=""):
    p = Path(p)
    # print(f" now in folder {p}")
    files = [h for h in p.iterdir() if h.is_file()]
    folders = [h for h in p.iterdir() if h.is_dir()]
    conversions = dict() # add (filename [str], success/failure [bool]) pairs
    import concurrent.futures
    executor = concurrent.futures.ProcessPoolExecutor(nthreads)
    futures = [executor.submit(iterate_threaded_helper, p, h, c, t) for h in files]
    # futures = [executor.submit(iterate_threaded_helper, args.p, h, str(args.c).replace('\'', '"'), args.t, str(args.i).lower() in ["true", "y", "t", "yes"])
    concurrent.futures.wait(futures)
    if i:
        for h in folders:
            # print(f"navigating to subfolder {p.joinpath(h)}")
            hh = str(h).split("\\")[-1]
            iterate_threaded(nthreads, p.joinpath(h), c.replace("(f)", f"{hh}\\(f)").replace("(fr)", f"{hh}\\(fr)").replace("(fe)", f"{hh}\\(fe)"), t, i)


def iterate(p, c, t, i, addToPath=""):
    p = Path(p)
    # print(f" now in folder {p}")
    files = [h for h in p.iterdir() if h.is_file()]
    folders = [h for h in p.iterdir() if h.is_dir()]
    conversions = dict() # add (filename [str], success/failure [bool]) pairs
    for h in files:
        path = p.joinpath(h)
        # print(p)
        if t != "*" and str(path.suffix.lower()) != t:
            continue
        a = c.replace("(f)", str(path)).replace("(fr)", str(path.stem)).replace("(fe)", str(path.suffix))
        # print(a)
        # exit()
        os.system(a)
        # import subprocess as sp
        # sp.call(a)
    if i:
        for h in folders:
            # print(f"navigating to subfolder {p.joinpath(h)}")
            hh = str(h).split("\\")[-1]
            # c.replace("(f)", f"(f)")
            iterate(p.joinpath(h), c.replace("(f)", f"{hh}/(f)").replace("(fr)", f"{hh}/(fr)").replace("(fe)", f"{hh}/(fe)"), t, i)
#        conversions = conversions | iterate(p.joinpath(h), c, t) # only works in python 3.9+
#        conversions = {**conversions, **iterate(p.joinpath(h), c, t)}

if __name__ == '__main__':
    # freeze_support()

    parser = argparse.ArgumentParser()
    parser.add_argument("-p", help="path to the workspace")
    parser.add_argument("-t", help="file type")
    parser.add_argument("-c", help="command to run with (f) for filename and (fr) & (fe) for the file root & extension")
    parser.add_argument("-threads", help="include to parallelize the commands")
    parser.add_argument("-i", help="include to iterate over subdirectories")
    args = parser.parse_args()

    chdir(args.p)

    if args.threads is not None and int(args.threads) > 1:
        nthreads = min(int(args.threads), 61)
        print(f'using {nthreads} threads')
        # exit()
        iterate_threaded(nthreads, args.p, str(args.c).replace('\'', '"'), args.t, str(args.i).lower() in ["true", "y", "t", "yes"])
    else:
        print('single thread')
        # exit()
        conversions = iterate(args.p, str(args.c).replace('\'', '"'), args.t, str(args.i).lower() in ["true", "y", "t", "yes"])

# python3 kss6/kCommandIterator.py -p "/Users/thekitchen/Desktop/photos/allie-usb/MyPhotos" -t ".jpg" -c "python3 /Users/thekitchen/Desktop/kss/kss-photo-tools/compress-photo.py -i '/Users/thekitchen/Desktop/photos/allie-usb/MyPhotos/(fr).jpg' -o '/Users/thekitchen/Desktop/photos/allie-usb/output/(fr).jpg' -r '.5:.5'" -i False
