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

import logging
import xgboost as xgb
from enum import Enum

LOGGER = logging.getLogger("[modin.xgboost]")


class RabitContextManager:
    def __init__(self, num_workers: int, host_ip):
        """Start Rabit tracker. The workers connect to this tracker to share
        their results."""

        self._num_workers = num_workers
        self.env = {"DMLC_NUM_WORKER": self._num_workers}
        self.rabit_tracker = xgb.RabitTracker(hostIP=host_ip, nslave=self._num_workers)

    def __enter__(self):
        self.env.update(self.rabit_tracker.slave_envs())
        self.rabit_tracker.start(self._num_workers)
        return self.env

    def __exit__(self, type, value, traceback):
        self.rabit_tracker.join()


class RabitContext:
    """Context to connect a worker to a rabit tracker"""

    def __init__(self, actor_ip, args):
        self.args = args
        self.args.append(("DMLC_TASK_ID=[modin.xgboost]:" + actor_ip).encode())

    def __enter__(self):
        xgb.rabit.init(self.args)
        LOGGER.info("-------------- rabit started ------------------")

    def __exit__(self, *args):
        xgb.rabit.finalize()
        LOGGER.info("-------------- rabit finished ------------------")


class DistributionType(Enum):
    """
    Enum for different modes of distribution the data.

    ``DistributionType.MIXED``  will distribute the data
    evenly between nodes on the cluster with minimal datatransfer.

    ``DistributionType.EVENLY`` will distribute the data
    evenly between nodes on the cluster.

    ``DistributionType.LOCALLY`` won't change the initial
    distribution of the data between nodes. It doesn't
    guarantee evenly nodes utilization.
    """

    MIXED = 0
    EVENLY = 1
    LOCALLY = 2
