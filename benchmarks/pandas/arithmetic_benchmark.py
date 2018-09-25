from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import os
import pandas as pd

from utils import time_logger


parser = argparse.ArgumentParser(description="arithmetic benchmark")
parser.add_argument("--path", dest="path", help="path to the csv data file")
parser.add_argument("--logfile", dest="logfile", help="path to the log file")
args = parser.parse_args()
file = args.path
file_size = os.path.getsize(file)

if not os.path.exists(os.path.split(args.logfile)[0]):
    os.makedirs(os.path.split(args.logfile)[0])

logging.basicConfig(filename=args.logfile, level=logging.INFO)

df = pd.read_csv(file)

with time_logger("Transpose: {}; Size: {} bytes".format(file, file_size)):
    df.T

with time_logger("Sum on axis=0: {}; Size: {} bytes".format(file, file_size)):
    df.sum()

with time_logger("Sum on axis=1: {}; Size: {} bytes".format(file, file_size)):
    df.sum(axis=1)

with time_logger("Median on axis=0: {}; Size: {} bytes".format(file, file_size)):
    df.median()

with time_logger("Median on axis=1: {}; Size: {} bytes".format(file, file_size)):
    df.median(axis=1)

with time_logger("nunique on axis=0: {}; Size: {} bytes".format(file, file_size)):
    df.nunique()

with time_logger("nunique on axis=1: {}; Size: {} bytes".format(file, file_size)):
    df.nunique(axis=1)

with time_logger("Sum UDF on axis=0: {}; Size: {} bytes".format(file, file_size)):
    df.apply(lambda df: df.sum())

with time_logger("Sum UDF on axis=1: {}; Size: {} bytes".format(file, file_size)):
    df.apply(lambda df: df.sum(), axis=1)
