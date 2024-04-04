"""
File to check performance in this PR
"""

import logging
import os
import time

from asv_bench.benchmarks.utils.common import generate_dataframe
from modin.config import BenchmarkMode, CpuCount, MinPartitionSize, NPartitions

BenchmarkMode.put(True)
# CpuCount.put(44)

logging.basicConfig(filename="error.log", filemode="w", level=logging.ERROR)


logger = logging.getLogger()

stratagies = [1, 2, 3]
# stratagies = ["pandas"]

iteration_number = 2

try:
    files = {
        stratagy: open(f"perf_{CpuCount.get()}_{stratagy}.csv", "w")
        for stratagy in stratagies
    }

    # write header
    for stratagy in stratagies:
        files[stratagy].write(
            ",".join(
                [
                    "Row parts",
                    "Column parts",
                    "Part size",
                    "Stratagy",
                    *[f"time_{i}" for i in range(iteration_number)],
                    "Min time",
                ]
            )
        )
        files[stratagy].write("\n")

    for part_size in [MinPartitionSize.get(), 100]:
        MinPartitionSize.put(part_size)
        for col_parts in range(1, 2 * CpuCount.get(), max(int(CpuCount.get() / 10), 1)):
            for row_parts in range(
                int(CpuCount.get() / 2),
                10 * CpuCount.get(),
                max(int(CpuCount.get() / 10), 1),
            ):
                NPartitions.put(max(col_parts, row_parts))
                for stratagy in stratagies:
                    if stratagy == "pandas":
                        df = generate_dataframe(
                            "int",
                            row_parts * part_size,
                            col_parts * part_size,
                            -100,
                            0,
                            impl="pandas",
                        )
                    else:
                        df = generate_dataframe(
                            "int",
                            row_parts * part_size,
                            col_parts * part_size,
                            -100,
                            0,
                            impl="modin",
                        )
                    os.environ["MY_STRATAGY"] = str(stratagy)
                    times = []
                    for _ in range(iteration_number):
                        t0 = time.perf_counter()
                        df.abs()
                        t1 = time.perf_counter()
                        times.append(t1 - t0)
                    files[stratagy].write(
                        ",".join(
                            [
                                str(x)
                                for x in [
                                    row_parts,
                                    col_parts,
                                    part_size,
                                    stratagy,
                                    *times,
                                    min(times),
                                ]
                            ]
                        )
                    )
                    files[stratagy].write("\n")
except Exception as ex:
    logger.exception(ex)
    print("ERROR!")  # noqa: T201
finally:
    for f in files.values():
        f.close()
