import os
import time
import ray
import modin.pandas as pd

ray.init(address="auto")
cpu_count = ray.cluster_resources()["CPU"]
assert cpu_count == 576, f"Expected 576 CPUs, but found {cpu_count}"

file_size = os.path.getsize("big_yellow.csv")


# get human readable file size
def sizeof_fmt(num, suffix="B"):
    for unit in ("", "K", "M", "G", "T"):
        if abs(num) < 1024.0:
            return f"{num:3.1f}{unit}{suffix}"
        num /= 1024.0
    return f"{num:.1f}P{suffix}"


print(f"File size is {sizeof_fmt(file_size)}")  # noqa: T201

t0 = time.perf_counter()
df = pd.read_csv("big_yellow.csv", quoting=3)
t1 = time.perf_counter()
print(f"read_csv time is {(t1 - t0):.3f}")  # noqa: T201

t0 = time.perf_counter()
count_result = df.count()
t1 = time.perf_counter()
print(f"count time is {(t1 - t0):.3f}")  # noqa: T201

t0 = time.perf_counter()
groupby_result = df.groupby("passenger_count").count()
t1 = time.perf_counter()
print(f"groupby time is {(t1 - t0):.3f}")  # noqa: T201

t0 = time.perf_counter()
apply_result = df.applymap(str)
t1 = time.perf_counter()
print(f"applymap time is {(t1 - t0):.3f}")  # noqa: T201
