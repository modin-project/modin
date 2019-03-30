from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import os
import pandas as pd

from utils import time_logger


parser = argparse.ArgumentParser(description="arithmetic benchmark")
parser.add_argument("--left", dest="left", help="path to the left csv data " "file")
parser.add_argument("--right", dest="right", help="path to the right csv data " "file")
parser.add_argument("--logfile", dest="logfile", help="path to the log file")
args = parser.parse_args()
file_left = args.left
file_size_left = os.path.getsize(file_left)

file_right = args.right
file_size_right = os.path.getsize(file_right)

if not os.path.exists(os.path.split(args.logfile)[0]):
    os.makedirs(os.path.split(args.logfile)[0])

logging.basicConfig(filename=args.logfile, level=logging.INFO)

df_left = pd.read_csv(file_left)
df_right = pd.read_csv(file_right)

with time_logger(
    "Inner Join: {} & {}; Left Size: {} bytes; Right Size: {} "
    "bytes".format(file_left, file_right, file_size_left, file_size_right)
):
    result = df_left.join(df_right, how="inner", lsuffix="left_")

with time_logger(
    "Outer Join: {} & {}; Left Size: {} bytes; Right Size: {} "
    "bytes".format(file_left, file_right, file_size_left, file_size_right)
):
    result = df_left.join(df_right, how="outer", lsuffix="left_")

with time_logger(
    "Inner Merge: {} & {}; Left Size: {} bytes; Right Size: {} "
    "bytes".format(file_left, file_right, file_size_left, file_size_right)
):
    result = df_left.merge(df_right, how="inner", left_index=True, right_index=True)

with time_logger(
    "Outer Merge: {} & {}; Left Size: {} bytes; Right Size: {} "
    "bytes".format(file_left, file_right, file_size_left, file_size_right)
):
    result = df_left.merge(df_right, how="outer", left_index=True, right_index=True)
