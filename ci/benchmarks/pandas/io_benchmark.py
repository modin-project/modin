from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import os
import pandas as pd

from utils import time_logger


parser = argparse.ArgumentParser(description="read_csv benchmark")
parser.add_argument("--path", dest="path", help="path to the csv file")
parser.add_argument("--logfile", dest="logfile", help="path to the log file")
args = parser.parse_args()
file = args.path
file_size = os.path.getsize(file)

if not os.path.exists(os.path.split(args.logfile)[0]):
    os.makedirs(os.path.split(args.logfile)[0])

logging.basicConfig(filename=args.logfile, level=logging.INFO)

with time_logger("Read csv file: {}; Size: {} bytes".format(file, file_size)):
    df = pd.read_csv(file)
