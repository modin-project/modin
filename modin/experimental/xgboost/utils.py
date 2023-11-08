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

"""Module holds classes for work with Rabit all-reduce context."""

import logging

import xgboost as xgb

LOGGER = logging.getLogger("[modin.xgboost]")


class RabitContextManager:
    """
    A manager class that controls lifecycle of `xgb.RabitTracker`.

    All workers that are used for distributed training will connect to
    Rabit Tracker stored in this class.

    Parameters
    ----------
    num_workers : int
        Number of workers of `self.rabit_tracker`.
    host_ip : str
        IP address of host that creates `self` object.
    """

    # TODO: Specify type of host_ip
    def __init__(self, num_workers: int, host_ip):
        self._num_workers = num_workers
        self.env = {"DMLC_NUM_WORKER": self._num_workers}
        self.rabit_tracker = xgb.RabitTracker(
            host_ip=host_ip, n_workers=self._num_workers
        )

    def __enter__(self):
        """
        Entry point of manager.

        Updates Rabit Tracker environment, starts `self.rabit_tracker`.

        Returns
        -------
        dict
            Dict with Rabit Tracker environment.
        """
        self.env.update(self.rabit_tracker.worker_envs())
        self.rabit_tracker.start(self._num_workers)
        return self.env

    # TODO: (type, value, traceback) -> *args
    def __exit__(self, type, value, traceback):
        """
        Exit point of manager.

        Finishes `self.rabit_tracker`.

        Parameters
        ----------
        type : exception type
            Type of exception, captured  by manager.
        value : Exception
            Exception value.
        traceback : TracebackType
            Traceback of exception.
        """
        self.rabit_tracker.join()


class RabitContext:
    """
    Context to connect a worker to a rabit tracker.

    Parameters
    ----------
    actor_rank : int
        Rank of actor, connected to this context.
    args : list
        List with environment variables for Rabit Tracker.
    """

    def __init__(self, actor_rank, args):
        self.args = args
        self.args.append(("DMLC_TASK_ID=[modin.xgboost]:" + str(actor_rank)).encode())

    def __enter__(self):
        """
        Entry point of context.

        Connects to Rabit Tracker.
        """
        xgb.rabit.init(self.args)
        LOGGER.info("-------------- rabit started ------------------")

    def __exit__(self, *args):
        """
        Exit point of context.

        Disconnects from Rabit Tracker.

        Parameters
        ----------
        *args : iterable
            Parameters for Exception capturing.
        """
        xgb.rabit.finalize()
        LOGGER.info("-------------- rabit finished ------------------")
