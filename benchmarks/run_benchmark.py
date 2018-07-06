import os
import argparse

parser = argparse.ArgumentParser(description='run benchmarks')
parser.add_argument('--N', dest='n', const=1, nargs='?', type=int,
                    help='number of test iterations to run')
args = parser.parse_args()
num_iterations = args.n

files = os.listdir("benchmarks/data")

for _ in range(num_iterations):
    for f in files:
        os.system("python benchmarks/arithmetic_benchmark.py "
                  "--path benchmarks/data/{} "
                  "--logfile benchmark-results/modin-arithmetic.log".format(f))
        os.system("python benchmarks/io_benchmark.py "
                  "--path benchmarks/data/{} "
                  "--logfile benchmark-results/modin-io.log".format(f))
