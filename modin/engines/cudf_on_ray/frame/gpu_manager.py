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

import ray
import cudf
import pandas
from .partition import cuDFOnRayFramePartition


@ray.remote(num_gpus=1)
class GPUManager(object):
    def __init__(self, gpu_id):
        self.key = 0
        self.cudf_dataframe_dict = {}
        self.gpu_id = gpu_id

    def length(self, key):
        return len(self.cudf_dataframe_dict[key])

    def width(self, key):
        return len(self.cudf_dataframe_dict[key].columns)

    def apply(self, key, func, **kwargs):
        df = self.cudf_dataframe_dict[key]
        cudf_dataframe = func(self.cudf_dataframe_dict[key], **kwargs)
        if isinstance(cudf_dataframe, cudf.Series):
            cudf_dataframe = cudf_dataframe.to_frame()
        self.key = self.key + 1
        self.cudf_dataframe_dict[self.key] = cudf_dataframe
        return self.key

    def apply_result_not_dataframe(self, key, func, **kwargs):
        return func(self.cudf_dataframe_dict[key], **kwargs)

    def apply_with_one_key_and_one_object_id(self, key, id, func, **kwargs):
        cudf_dataframe_1 = self.cudf_dataframe_dict[key]
        cudf_dataframe_2 = id
        cudf_dataframe = func(cudf_dataframe_1, cudf_dataframe_2)
        if isinstance(cudf_dataframe, cudf.Series):
            cudf_dataframe = cudf_dataframe.to_frame()
        self.key = self.key + 1
        self.cudf_dataframe_dict[self.key] = cudf_dataframe
        return self.key

    def apply_with_two_keys(self, key_1, key_2, func, **kwargs):
        cudf_dataframe = func(self.cudf_dataframe_dict[key_1], self.cudf_dataframe_dict[key_2])
        if isinstance(cudf_dataframe, cudf.Series):
            cudf_dataframe = cudf_dataframe.to_frame()
        self.key = self.key + 1
        self.cudf_dataframe_dict[self.key] = cudf_dataframe
        return self.key

    # Since we are using, a row partitioning scheme, there is no need for us to go through object
    # store. Work directly with keys.
    def reduce_key_list(self, keys, func, **kwargs):
        df = self.cudf_dataframe_dict[keys[0]]
        for i in range(1, len(keys)):
            df = cudf.concat([df, self.cudf_dataframe_dict[keys[i]]], axis=1)
        result = func(df)
        if isinstance(result, cudf.Series):
            result = result.to_frame()
        self.key = self.key + 1
        self.cudf_dataframe_dict[self.key] = result
        return self.key

    def brute_force_merge(self, key, right_partitions, func):
        # This avoids memory leaking on merge.
        def merge_reduce_worker(left, right, func):
            if isinstance(right, int):
                return func(left, self.cudf_dataframe_dict[right])

            id = ray.get(right)
            if isinstance(id, int):
                return func(left, self.cudf_dataframe_dict[id])
            else:
                return func(left, id)

        cudf_dataframe_1 = self.cudf_dataframe_dict[key]
        cudf_dataframes = []
        for id in right_partitions:
            cudf_dataframes.append(merge_reduce_worker(cudf_dataframe_1, id, func))
        self.key = self.key + 1
        self.cudf_dataframe_dict[self.key] = cudf.concat([i for i in cudf_dataframes])
        return self.key

    def reduce(self, cudf_dataframe_object_ids, axis, func, **kwargs):
        cudf_dataframes = ray.get(cudf_dataframe_object_ids)
        dataframe = cudf.concat(cudf_dataframes, axis=axis)
        print(f"reduce dataframe={dataframe}")
        # FIXME (kvu35): Hacky solution. Not sure if this logic should be this
        # low in the stack.
        # cudf.concat for some reason resets the names of the multiindex, so
        # we have to make sure to reassign it. This messes up apis like
        # groupby and reset_index
        if isinstance(dataframe.index, cudf.core.multiindex.MultiIndex):
            dataframe.index = cudf.concat([df.index for df in cudf_dataframes])
        # some function only works on series so we convert
        result = func(dataframe, **kwargs)
        if isinstance(result, cudf.Series):
            result = result.to_frame()
        self.key = self.key + 1
        self.cudf_dataframe_dict[self.key] = result
        return self.key

    def put(self, dataframe):
        if isinstance(dataframe, pandas.Series):
            dataframe = dataframe.to_frame()
        self.key = self.key + 1
        self.cudf_dataframe_dict[self.key] = cudf.from_pandas(dataframe)
        return self.key

    def get(self, key):
        return self.cudf_dataframe_dict[key]

    def get_object_id(self, key):
        return self.cudf_dataframe_dict[key]

    def to_pandas(self, key):
        return self.cudf_dataframe_dict[key].to_pandas()

    def free(self, key):
        if key in self.cudf_dataframe_dict:
            del self.cudf_dataframe_dict[key]

    def get_id(self):
        return self.gpu_id

    def apply_with_key_oid_list(self, key, oids, func, join_type="join", **kwargs):
        join_func = cudf.DataFrame.join
        if join_type == "concat":
            join_func = lambda x, y: cudf.concat([x, y])
        oids = ray.get(oids)
        cudf_df_1 = self.cudf_dataframe_dict[key]
        cudf_df_2 = oids[0] if len(oids) >= 1 else None
        for i in range(1, len(oids)):
            cudf_df_2 = join_func(cudf_df_2, oids[i])
        result = func(cudf_df_1, cudf_df_2)
        self.key = self.key + 1
        self.cudf_dataframe_dict[self.key] = result
        return self.key
