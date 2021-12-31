import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="input path")
parser.add_argument("-o", help="output path")
parser.add_argument("-r", help="args & filters")
args = parser.parse_args()

import os
f = "/".join(args.o.split("/")[:-1])
if not os.path.exists(f):
    os.mkdir(f)

# os.system(f"ffmpeg -i \"{args.i}\"")
print(f"ffmpeg -i \"{args.i}\" {args.r} \"{args.o}\"")
