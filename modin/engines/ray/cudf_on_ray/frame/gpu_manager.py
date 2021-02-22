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


@ray.remote(num_gpus=1)
class GPUManager(object):
    def __init__(self, gpu_id):
        self.key = 0
        self.cudf_dataframe_dict = {}
        self.gpu_id = gpu_id

    ## TODO(#45): Merge apply and apply_non_persistent
    def apply_non_persistent(self, first, other, func, **kwargs):
        df1 = self.cudf_dataframe_dict[first]
        df2 = self.cudf_dataframe_dict[other] if other else None
        if not df2:
            result = func(df1, **kwargs)
        else:
            result = func(df1, df2, **kwargs)
        return result

    def apply(self, first, other, func, **kwargs):
        df1 = self.cudf_dataframe_dict[first]
        if not other:
            result = func(df1, **kwargs)
            return self.store_new_df(result)
        if not isinstance(other, int):
            assert isinstance(other, ray.ObjectRef)
            df2 = ray.get(other)
        else:
            df2 = self.cudf_dataframe_dict[other]
        result = func(df1, df2, **kwargs)
        return self.store_new_df(result)

    def reduce(self, first, others, func, axis=0, **kwargs):
        join_func = (
            cudf.DataFrame.join if not axis else lambda x, y: cudf.concat([x, y])
        )
        if not isinstance(others[0], int):
            other_dfs = ray.get(others)
        else:
            other_dfs = [self.cudf_dataframe_dict[i] for i in others]
        df1 = self.cudf_dataframe_dict[first]
        df2 = others[0] if len(others) >= 1 else None
        for i in range(1, len(others)):
            df2 = join_func(df2, other_dfs[i])
        result = func(df1, df2, **kwargs)
        return self.store_new_df(result)

    def store_new_df(self, df):
        self.key += 1
        self.cudf_dataframe_dict[self.key] = df
        return self.key

    def free(self, key):
        if key in self.cudf_dataframe_dict:
            del self.cudf_dataframe_dict[key]

    def get_id(self):
        return self.gpu_id

    def get_oid(self, key):
        return self.cudf_dataframe_dict[key]

    def put(self, pandas_df):
        if isinstance(pandas_df, pandas.Series):
            pandas_df = pandas_df.to_frame()
        return self.store_new_df(cudf.from_pandas(pandas_df))
