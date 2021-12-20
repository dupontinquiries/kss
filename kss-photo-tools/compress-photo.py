from PIL import Image
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-i", help="input path")
parser.add_argument("-o", help="output path")
parser.add_argument("-r", help="scale-x:scale-y")
args = parser.parse_args()

im = Image.open(args.i)

a,b = args.r.split(":")

a = float(a)
b = float(b)

newsize = (int(im.size[0] * a), int(im.size[1] * b))

n = im.resize(newsize)

# n.show()

n.save(args.o, "JPEG") # JPEG2000