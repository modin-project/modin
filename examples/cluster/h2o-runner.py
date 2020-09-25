# Licensed to Modin Development Team under one or more contributor license agreements.
# See the NOTICE file distributed with this work for additional information regarding
# copyright ownership.  The Modin Development Team licenses this file to you under the
# Apache License, Version 2.0 (the "License"); you may not use this file except in
# compliance with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software distributed under
# the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific language
# governing permissions and limitations under the License.


# pip install git+https://github.com/intel-ai/ibis.git@develop
# pip install braceexpand

# NOTE: expects https://github.com/intel-ai/omniscripts checked out and in PYTHONPATH

# the following import turns on experimental mode in Modin,
# including enabling running things in remote cloud
import modin.experimental.pandas as pd  # noqa: F401
from modin.experimental.cloud import create_cluster

from h2o import run_benchmark

test_cluster = create_cluster(
    "aws",
    "aws_credentials",
    cluster_name="rayscale-test",
    region="eu-north-1",
    zone="eu-north-1b",
    image="ami-00e1e82d7d4ca80d3",
)
with test_cluster:
    parameters = {
        "no_pandas": False,
        "pandas_mode": "Modin_on_ray",
        "ray_tmpdir": "/tmp",
        "ray_memory": 1024 * 1024 * 1024,
        "extended_functionality": False,
    }

    # G1... - for groupby queries; J1... - for join queries;
    # Additional required files inside h2o-data folder:
    # - J1_1e6_1e0_0_0.csv
    # - J1_1e6_1e3_0_0.csv
    # - J1_1e6_1e6_0_0.csv
    for data_file in ["G1_5e5_1e2_0_0.csv", "J1_1e6_NA_0_0.csv"]:
        parameters["data_file"] = f"https://modin-datasets.s3.amazonaws.com/h2o/{data_file}"
        run_benchmark(parameters)
