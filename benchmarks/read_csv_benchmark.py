from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import os
import ray
import modin.pandas as pd

from utils import time_logger


logging.basicConfig(filename='benchmarks.log', level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Modin read_csv Benchmark')
parser.add_argument('--path', dest='path', help='path to the csv file')
args = parser.parse_args()
file = args.path
file_size = os.path.getsize(file)

with time_logger("Read csv file: {}; Size: {} bytes".format(file, file_size)):
    df = pd.read_csv(file)
    blocks = df._block_partitions.flatten().tolist()
    ray.wait(blocks, len(blocks))
