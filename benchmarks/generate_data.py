from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import modin.pandas as pd
import numpy as np
import os

num_rows = [100, 10000, 100000, 150000, 200000, 350000, 500000]
num_cols = [1000]

path_to_data = "benchmarks/data/"
if not os.path.exists(path_to_data):
    os.makedirs(path_to_data)

for r in num_rows:
    for c in num_cols:
        df = pd.DataFrame(np.random.randint(0, 100, size=(r, c)))
        df.to_csv(path_to_data + "test-data-{}-{}.csv".format(r, c))

# Files for multi df tests
num_rows = [100, 1000, 100000, 1000000]
num_cols = [1000]

path_to_data = "benchmarks/data/multi/"
if not os.path.exists(path_to_data):
    os.makedirs(path_to_data)

for r in num_rows:
    for c in num_cols:
        df = pd.DataFrame(np.random.randint(0, 100, size=(r, c)))
        df.to_csv(path_to_data + "test-data-{}-{}.csv".format(r, c))
