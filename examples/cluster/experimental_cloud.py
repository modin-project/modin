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

"""
This is a very basic sample script for running things remotely.
It requires `aws_credentials` file to be present next to it.

On credentials file format see https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html#cli-configure-files-where
"""

import os
import logging
logging.basicConfig(format='%(asctime)s %(message)s')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)

from modin.experimental.cloud import Provider, cluster, get_connection


aws = Provider(Provider.AWS, os.path.join(os.path.abspath(os.path.dirname(__file__)), 'aws_credentials'), 'us-west-1')
with cluster(aws, cluster_name='rayscale-test') as c:
    conn = get_connection()
    modin = conn.modules.modin
    print(modin)
    print(type(modin))
    import pdb;pdb.set_trace()
