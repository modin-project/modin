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

import pandas

from modin.engines.base.frame.partition import BaseFramePartition
import pyarrow

import ray


class OmnisciOnRayFramePartition(BaseFramePartition):
    def __init__(
        self, object_id=None, frame_id=None, arrow_table=None, length=None, width=None
    ):
        assert type(object_id) is ray.ObjectID

        self.oid = object_id
        self.frame_id = frame_id
        self.arrow_table = arrow_table
        self._length_cache = length
        self._width_cache = width

    def to_pandas(self):
        obj = self.get()
        if isinstance(obj, (pandas.DataFrame, pandas.Series)):
            return obj
        assert isinstance(obj, pyarrow.Table)
        return obj.to_pandas()

    def get(self):
        if self.arrow_table is not None:
            return self.arrow_table
        return ray.get(self.oid)

    @classmethod
    def put(cls, obj):
        return OmnisciOnRayFramePartition(
            object_id=ray.put(obj),
            length=len(obj.index),
            width=len(obj.columns),
        )

    @classmethod
    def put_arrow(cls, obj):
        return OmnisciOnRayFramePartition(
            object_id=ray.put(None),
            arrow_table=obj,
            length=len(obj),
            width=len(obj.columns),
        )
