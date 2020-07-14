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

import os
os.environ["MODIN_ENGINE"] = "python"
os.environ['MODIN_EXPERIMENTAL'] = 'True'

# logging for Ray
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.INFO)

import modin.pandas as pd

from mortgage import run_benchmark
from mortgage.mortgage_pandas import etl_pandas

from modin.experimental.cloud import Provider, Cluster, get_connection

aws = Provider(Provider.AWS, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'aws_credentials'), 'us-west-1')
cl = Cluster(aws, cluster_name='rayscale-test')

with cl:
    import rpyc
    conn: rpyc.ClassicService = get_connection()
    np = conn.modules["numpy"]
    run_benchmark.__globals__["pd"] = pd
    etl_pandas.__globals__["pd"] = pd
    etl_pandas.__globals__["np"] = np

    parameters = {}
    #parameters["data_file"] = "/home/ubuntu/bench_data"
    parameters["data_file"] = "https://modin-datasets.s3.amazonaws.com/mortgage"
    parameters["dfiles_num"] = 1
    parameters["no_ml"] = False
    parameters["validation"] = False
    parameters["no_ibis"] = True
    parameters["no_pandas"] = False
    parameters["pandas_mode"] = "Modin_on_ray"

    run_benchmark(parameters)
