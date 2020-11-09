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


class PartitionUnwrapper(object):
    @classmethod
    def unwrap(cls, query_compiler, axis, engine=None):
        if engine is not None and engine.lower() not in type(query_compiler).__name__:
            raise ValueError("Engine does not match query compiler object")
        mgr = query_compiler._modin_frame._frame_mgr_cls
        if axis == 0:
            partitions = mgr.row_partitions(query_compiler._modin_frame._partitions)
        elif axis == 1:
            partitions = mgr.column_partitions(query_compiler._modin_frame._partitions)
        else:
            raise ValueError("Axis attribute provided not yet supported")
        return [part.coalesce().unwrap(squeeze=True) for part in partitions]


def unwrap_row_partitions(api_layer_object, engine=None):
    if not hasattr(api_layer_object, "_query_compiler"):
        raise ValueError("Only API Layer objects may be passed in here.")
    return PartitionUnwrapper.unwrap(
        api_layer_object._query_compiler, axis=0, engine=engine
    )


def unwrap_column_partitions(api_layer_object, engine=None):
    if not hasattr(api_layer_object, "_query_compiler"):
        raise ValueError("Only API Layer objects may be passed in here.")
    return PartitionUnwrapper.unwrap(
        api_layer_object._query_compiler, axis=1, engine=engine
    )
