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

import numpy as np

from modin.engines.ray.generic.frame.partition_manager import RayFrameManager
from .axis_partition import (
    OmnisciOnRayFrameColumnPartition,
    OmnisciOnRayFrameRowPartition,
)
from .partition import OmnisciOnRayFramePartition
from .omnisci_worker import OmnisciServer
from .calcite_builder import CalciteBuilder
from .calcite_serializer import CalciteSerializer

import pyarrow
import pandas
import os


class OmnisciOnRayFrameManager(RayFrameManager):
    """This method implements the interface in `BaseFrameManager`."""

    # This object uses RayRemotePartition objects as the underlying store.
    _partition_class = OmnisciOnRayFramePartition
    _column_partitions_class = OmnisciOnRayFrameColumnPartition
    _row_partition_class = OmnisciOnRayFrameRowPartition

    @classmethod
    def _compute_num_partitions(cls):
        """Currently, we don't handle partitioned frames for OmniSci engine.
        Since we support a single node mode only, allow OmniSci perform
        partitioning by itself.

        :return:
        """
        return 1

    @classmethod
    def from_arrow(cls, at, return_dims=False):
        put_func = cls._partition_class.put_arrow

        parts = [[put_func(at)]]

        if not return_dims:
            return np.array(parts)
        else:
            row_lengths = [at.num_rows]
            col_widths = [at.num_columns]
            return np.array(parts), row_lengths, col_widths

    @classmethod
    def run_exec_plan(cls, plan, index_cols, dtypes, columns):
        # TODO: this plan is supposed to be executed remotely using Ray.
        # For now OmniSci engine support only a single node cluster.
        # Therefore remote execution is not necessary and will be added
        # later.
        omniSession = OmnisciServer()

        # First step is to make sure all partitions are in OmniSci.
        frames = plan.collect_frames()
        for frame in frames:
            if frame._partitions.size != 1:
                raise NotImplementedError(
                    "OmnisciOnRay engine doesn't suport partitioned frames"
                )
            for p in frame._partitions.flatten():
                if p.frame_id is None:
                    obj = p.get()
                    if isinstance(obj, (pandas.DataFrame, pandas.Series)):
                        p.frame_id = omniSession.put_pandas_to_omnisci(obj)
                    else:
                        assert isinstance(obj, pyarrow.Table)
                        p.frame_id = omniSession.put_arrow_to_omnisci(obj)

        calcite_plan = CalciteBuilder().build(plan)
        calcite_json = CalciteSerializer().serialize(calcite_plan)

        cmd_prefix = "execute relalg "

        use_calcite_env = os.environ.get("MODIN_USE_CALCITE")
        use_calcite = use_calcite_env is not None and use_calcite_env.lower() == "true"

        if use_calcite:
            cmd_prefix = "execute calcite "

        curs = omniSession.executeRA(cmd_prefix + calcite_json)
        assert curs
        rb = curs.getArrowRecordBatch()
        assert rb
        at = pyarrow.Table.from_batches([rb])

        res = np.empty((1, 1), dtype=np.dtype(object))
        # workaround for https://github.com/modin-project/modin/issues/1851
        if use_calcite:
            at = at.rename_columns(["F_" + str(c) for c in columns])
        res[0][0] = cls._partition_class.put_arrow(at)

        return res

    @classmethod
    def _names_from_index_cols(cls, cols):
        if len(cols) == 1:
            return cls._name_from_index_col(cols[0])
        return [cls._name_from_index_col(n) for n in cols]

    @classmethod
    def _name_from_index_col(cls, col):
        if col.startswith("__index__"):
            return None
        return col

    @classmethod
    def _maybe_scalar(cls, lst):
        if len(lst) == 1:
            return lst[0]
        return lst
