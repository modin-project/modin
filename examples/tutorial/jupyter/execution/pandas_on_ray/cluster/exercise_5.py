import time

import ray

import modin.pandas as pd

ray.init(address="auto")
cpu_count = ray.cluster_resources()["CPU"]
assert cpu_count == 576, f"Expected 576 CPUs, but found {cpu_count}"

file_path = "big_yellow.csv"

t0 = time.perf_counter()

df = pd.read_csv(file_path, quoting=3)
df_count = df.count()
df_groupby_count = df.groupby("passenger_count").count()
df_map = df.map(str)

t1 = time.perf_counter()
print(f"Full script time is {(t1 - t0):.3f}")  # noqa: T201
