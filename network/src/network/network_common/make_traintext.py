import os, sys
import argparse

from yaml import parse

parser = argparse.ArgumentParser(description="fdf")

parser.add_argument("--folder")
args = parser.parse_args()
path = os.path.join(args.folder, "images")


files = os.listdir(path)

f = open(os.path.join(args.folder, "train.txt"), "w")
for fi in files:
    f.write(fi+"\n")


