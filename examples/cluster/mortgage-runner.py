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


# pip install git+https://github.com/intel-go/ibis.git@develop

# NOTE: expects https://github.com/intel-go/omniscripts/tree/modin-rpyc-test checked out and in PYTHONPATH

# the following import turns on experimental mode in Modin,
# including enabling running things in remote cloud
import modin.experimental.pandas as pd  # noqa: F401
from modin.experimental.cloud import create_cluster, get_connection

from mortgage import run_benchmark
from mortgage.mortgage_pandas import etl_pandas

cl = create_cluster("aws", "aws_credentials", cluster_name="rayscale-test")

with cl:
    conn = get_connection()
    np = conn.modules["numpy"]
    etl_pandas.__globals__["np"] = np

    parameters = {
        "data_file": "https://modin-datasets.s3.amazonaws.com/mortgage",
        "dfiles_num": 1,
        "no_ml": True,
        "validation": False,
        "no_ibis": True,
        "no_pandas": False,
        "pandas_mode": "Modin_on_ray",
    }

    run_benchmark(parameters)
