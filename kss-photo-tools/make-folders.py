

import argparse
import os
from pathlib import Path

def iterate(i, o):
    pi = Path(i)
    # print(f'o = {o}')
    # po = Path(o)
    # print(f'po = {po}')
    for h in [h for h in pi.iterdir() if h.is_dir()]:
        # print(f'h = {h}')
        hh = str(h).split("/")[-1]
        if not os.path.exists( o + "/" + hh ):
            os.mkdir( o + "/" + hh )
        # print(f'po/hh = {str(po) + "--" + hh}')
        # print(f'hh = {hh}')
        # print( str(po.joinpath(hh)) )
        # print(f'po+hh = {po.joinpath(hh)}')
        iterate(str(pi.joinpath(h)), o + "/" + hh)

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="top directory to mimic")
parser.add_argument("-o", help="location to mimic top directory")
args = parser.parse_args()

conversions = iterate(args.i, args.o)
