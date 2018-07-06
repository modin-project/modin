import os

files = os.listdir("benchmarks/data")

for f in files:
    os.system("python benchmarks/arithmetic_benchmark.py "
              "--path benchmarks/data/{} "
              "--logfile benchmark-results/aaamodin-arithmetic.log".format(f))
    os.system("python benchmarks/io_benchmark.py "
              "--path benchmarks/data/{} "
              "--logfile benchmark-results/aaamodin-io.log".format(f))
