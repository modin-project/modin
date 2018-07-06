import os

files = os.listdir("benchmarks/data")

for f in files:
    os.system("python benchmarks/arithmetic_benchmark.py "
              "--path benchmarks/data/{} "
              "--logfile benchmark-results/modin-arithmetic.log".format(f))
    os.system("python benchmarks/io_benchmark.py "
              "--path benchmarks/data/{} "
              "--logfile benchmark-results/modin-io.log".format(f))
