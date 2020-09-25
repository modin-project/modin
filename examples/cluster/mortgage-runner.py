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

from mortgage import run_benchmark

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
        "data_file": "https://modin-datasets.s3.amazonaws.com/mortgage",
        # "data_file": "s3://modin-datasets/mortgage",
        "dfiles_num": 1,
        "no_ml": True,
        "validation": False,
        "no_ibis": True,
        "no_pandas": False,
        "pandas_mode": "Modin_on_ray",
        "ray_tmpdir": "/tmp",
        "ray_memory": 1024 * 1024 * 1024,
    }

    run_benchmark(parameters)
