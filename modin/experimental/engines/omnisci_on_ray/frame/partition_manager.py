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
from modin.error_message import ErrorMessage
from modin import __execution_engine__
from .omnisci_worker import OmnisciServer
from .calcite_builder import CalciteBuilder
from .calcite_serializer import CalciteSerializer

import json

if __execution_engine__ == "Ray":
    import ray


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
    def run_exec_plan(cls, plan, index_cols, dtypes):
        # TODO: this plan is supposed to be executed remotely using Ray.
        # For now OmniSci engine support only a single node cluster.
        # Therefore remote execution is not necessary and will be added
        # later.

        print("Executing DF plan:")
        plan.dump(">")

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
                    df = ray.get(p.oid)
                    # Currently DataFrame loader uploads only columns
                    # and ignores index. So, move index to columns and
                    # set proper column names.
                    if frame._index_cols is not None:
                        df.reset_index(inplace=True)
                        df.columns = frame._table_cols
                    p.frame_id = omniSession.put_pandas_to_omnisci(df)

        calcite_plan = CalciteBuilder().build(plan)
        calcite_json = CalciteSerializer().serialize(calcite_plan)

        curs = omniSession.executeRA("execute relalg "+calcite_json) # remove prefix
        rb = curs.getArrowRecordBatch()
        assert rb
        df = rb.to_pandas()

        # Currently boolean columns are loaded as integer
        # series for some reason. Fix it here for now.
        # Using Arrow should solve the problem later.
        for col in dtypes.index:
            if dtypes[col] == "bool":
                df[col] = df[col].astype("bool")
            elif dtypes[col].name == "category":
                df[col] = df[col].astype("category")

        if index_cols is not None:
            df = df.set_index(index_cols)
            df.index.rename(cls._names_from_index_cols(index_cols), inplace=True)

        # print("Execution result:")
        # print(df)
        # print(df.dtypes)

        res = np.empty((1, 1), dtype=np.dtype(object))
        res[0][0] = cls._partition_class.put(df)

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
