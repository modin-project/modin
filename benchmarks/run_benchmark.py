from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import argparse
from subprocess import Popen, DEVNULL, TimeoutExpired

parser = argparse.ArgumentParser(description="run benchmarks")
parser.add_argument(
    "--N",
    dest="n",
    default=1,
    nargs="?",
    type=int,
    help="number of test iterations to run",
)
args = parser.parse_args()
num_iterations = args.n

files = os.listdir("benchmarks/data")

timeout = 60 * 20
for _ in range(num_iterations):
    for f in files:
        p = Popen(
            [
                "python",
                "benchmarks/arithmetic_benchmark.py",
                "--path",
                "benchmarks/data/{}".format(f),
                "--logfile",
                "benchmark-results/modin-arithmetic.log",
            ],
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        try:
            p.wait(timeout)
        except TimeoutExpired:
            p.kill()

    for f in files:
        p = Popen(
            [
                "python",
                "benchmarks/groupby_benchmark.py",
                "--path",
                "benchmarks/data/{}".format(f),
                "--logfile",
                "benchmark-results/modin-groupby.log",
            ],
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        try:
            p.wait(timeout)
        except TimeoutExpired:
            p.kill()

    for f in files:
        p = Popen(
            [
                "python",
                "benchmarks/io_benchmark.py",
                "--path",
                "benchmarks/data/{}".format(f),
                "--logfile",
                "benchmark-results/modin-io.log",
            ],
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        try:
            p.wait(timeout)
        except TimeoutExpired:
            p.kill()

    for f in files:
        p = Popen(
            [
                "python",
                "benchmarks/df_op_benchmark.py",
                "--path",
                "benchmarks/data/{}".format(f),
                "--logfile",
                "benchmark-results/modin-df-op.log",
            ],
            stdout=DEVNULL,
            stderr=DEVNULL,
        )
        try:
            p.wait(timeout)
        except TimeoutExpired:
            p.kill()

    multi_df_files = os.listdir("benchmarks/data/multi")

    for f in files:
        for g in multi_df_files:
            p = Popen(
                [
                    "python",
                    "benchmarks/join_merge_benchmark.py",
                    "--left",
                    "benchmarks/data/{}".format(f),
                    "--right",
                    "benchmarks/data/{}".format(g),
                    "--logfile",
                    "benchmark-results/modin-join-merge.log",
                ],
                stdout=DEVNULL,
                stderr=DEVNULL,
            )
            try:
                p.wait(timeout)
            except TimeoutExpired:
                p.kill()
