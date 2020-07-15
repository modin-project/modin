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
from modin.data_management.utils import length_fn_pandas, width_fn_pandas
from modin.engines.ray.utils import handle_ray_task_error
from modin import __execution_engine__
from .omnisci_worker import OmnisciServer

if __execution_engine__ == "Ray":
    import ray
    from ray.worker import RayTaskError


class OmnisciOnRayFramePartition(BaseFramePartition):
    def __init__(self, object_id=None, frame_id=None, arrow_slice=None, length=None, width=None):
        assert type(object_id) is ray.ObjectID

        self.oid = object_id
        self.frame_id = frame_id
        self.arrow_slice = arrow_slice
        self._length_cache = length
        self._width_cache = width

    def to_pandas(self):
        print ("Warning: switching to Pandas DataFrame..")
        dataframe = self.get()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series
        return dataframe

    def get(self):
        return ray.get(self.oid)

    @classmethod
    def put(cls, obj):
        return OmnisciOnRayFramePartition(
            object_id=ray.put(obj),
            # frame_id = None,
            length=len(obj.index),
            width=len(obj.columns),
        )

    @classmethod
    def put_arrow(cls, obj):
        return OmnisciOnRayFramePartition(
            object_id=ray.put(obj),
            frame_id=OmnisciServer().put_arrow_to_omnisci(obj),  # TODO: make materialization later?
            arrow_slice=obj,                                     # TODO question of life time when loaded in omnisci dbe
            length=len(obj),
            width=len(obj.columns),
        )
