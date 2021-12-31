

import argparse
import os
from pathlib import Path

def iterate(i, o):
    pi = Path(i)
    print(f'o = {o}')
    po = Path(o)
    for h in [h for h in pi.iterdir() if h.is_dir()]:
        hh = str(h).split("\\")[-1]
        if not os.path.exists(str(h)):
            os.mkdir(str(h))
        iterate(str(pi.joinpath(h)), str(po.joinpath(h)))

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="top directory to mimic")
parser.add_argument("-o", help="location to mimic top directory")
args = parser.parse_args()

conversions = iterate(args.i, args.o)
