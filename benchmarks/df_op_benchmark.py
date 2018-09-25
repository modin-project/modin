from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import argparse
import ray
import os
import modin.pandas as pd

from utils import time_logger
import numpy as np

parser = argparse.ArgumentParser(description="arithmetic benchmark")
parser.add_argument("--path", dest="path", help="path to the csv data file")
parser.add_argument("--logfile", dest="logfile", help="path to the log file")
args = parser.parse_args()
file = args.path
file_size = os.path.getsize(file)

logging.basicConfig(filename=args.logfile, level=logging.INFO)

df = pd.read_csv(file)
blocks = df._block_partitions.flatten().tolist()
ray.wait(blocks, len(blocks))

num_rows, num_cols = df.shape
new_row = np.random.randint(0, 100, size=num_cols)
new_col = np.random.randint(0, 100, size=num_rows)


def rand_row_loc():
    return np.random.randint(0, num_rows)


def rand_col_loc():
    return np.random.randint(0, num_cols)


# row/col r/w
with time_logger("read a column: {}; Size: {} bytes".format(file, file_size)):
    df.iloc[:, rand_col_loc()]

with time_logger("read a row: {}; Size: {} bytes".format(file, file_size)):
    df.iloc[rand_row_loc(), :]

with time_logger("write a column: {}; Size: {} bytes".format(file, file_size)):
    df.iloc[:, rand_col_loc()] = new_col

with time_logger("write a row: {}; Size: {} bytes".format(file, file_size)):
    df.iloc[rand_row_loc(), :] = new_row

# element r/w

with time_logger("read an element: {}; Size: {} bytes".format(file, file_size)):
    df.iloc[rand_row_loc(), rand_col_loc()]

with time_logger("write an element: {}; Size: {} bytes".format(file, file_size)):
    df.iloc[rand_row_loc(), rand_col_loc()] = np.random.randint(0, 100)

# appending
with time_logger("append a row: {}; Size: {} bytes".format(file, file_size)):
    df.append(pd.Series(new_row), ignore_index=True)

with time_logger("append a column: {}; Size: {} bytes".format(file, file_size)):
    df["new"] = new_col
