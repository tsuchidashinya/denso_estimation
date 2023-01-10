import os
import argparse
from util import util


parser = argparse.ArgumentParser(description="fdf")

parser.add_argument("--folder")
args = parser.parse_args()
path = os.path.join(args.folder, "train.txt")
files = open(path).read().splitlines()
for file in files:
    file = util.exclude_ext_str(file)
    file_path = os.path.join(args.folder, "labels", file + ".txt")
    
    lines = open(file_path).read().splitlines()
    if (len(lines) == 0):
        print(file_path)
    for line in lines:
        label = line.split()
        if len(label) != 5:
            print(len(label))
            print(file_path)

