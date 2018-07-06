from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse

parser = argparse.ArgumentParser(description='run benchmarks')
parser.add_argument('--N', dest='n', const=1, nargs='?', type=int,
                    help='number of test iterations to run')
parser.add_argument('--multi', dest='multi', const=False, nargs='?', type=bool,
                    help="performing an operation that involves multiple "
                         "dataframes")
args = parser.parse_args()
multi = args.multi
num_iterations = args.n

files = os.listdir("benchmarks/data")

if not multi:
    for _ in range(num_iterations):
        for f in files:
            os.system("python benchmarks/arithmetic_benchmark.py "
                      "--path benchmarks/data/{} "
                      "--logfile benchmark-results/modin-arithmetic.log".format(f))
            os.system("python benchmarks/io_benchmark.py "
                      "--path benchmarks/data/{} "
                      "--logfile benchmark-results/modin-io.log".format(f))

else:
    for _ in range(num_iterations):
        for f in files:
            for g in files:
                if f == g:
                    continue
                os.system("python benchmarks/join_merge_benchmark.py "
                          "--left benchmarks/data/{} "
                          "--right benchmarks/data/{} "
                          "--logfile benchmark-results/modin-join-merge.log".format(f, g))
