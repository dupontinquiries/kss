import argparse
import zipfile

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="input path")
parser.add_argument("-o", help="output path")
args = parser.parse_args()

with zipfile.ZipFile(args.i, 'r') as hh:
    hh.extractall(args.o)
