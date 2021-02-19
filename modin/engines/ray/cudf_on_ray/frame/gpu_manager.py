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


# ray remote -- parallelizes a class/function
# do you call it as x = GPUManager(1).remote()?
# it would be x = ray.get(GPUManager.remote(gpu_id))
@ray.remote(num_gpus=1) 
class GPUManager(object):

    # constructor/initializer method, takes gpu_id.
    def __init__(self, gpu_id):
        self.key = 0
        self.cudf_dataframe_dict = {}  # what does this do? what is the purpose of this dictionary?
        self.gpu_id = gpu_id

    
    # df1 = self.cudf_dataframe_dict[first] -- value for first key.
    # dictionaries are indexed like arrays in Java. So dict[key] gets you the value for that key
    # so we pass in two keys, a func. we run func(df1, df2, **kwargs), and return the result.
    def apply_non_persistent(self, first, other, func, **kwargs):
        df1 = self.cudf_dataframe_dict[first]
        df2 = self.cudf_dataframe_dict[other] if other else None # if other is not passed in, None.
        if not df2:
            result = func(df1, **kwargs)
        else:
            result = func(df1, df2, **kwargs)
        return result

    # get df1 (the value of dataframe_dict)
    # if other is not an int, then check if other is an objectRef, 
    # if it is, get the value of the object
    # else, just get the value from the dictionary for other.
    # run func(df1, df2, **kwargs), store in result.
    # run store_new_df(result).
    def apply(self, first, other, func, **kwargs):
        df1 = self.cudf_dataframe_dict[first]
        if not isinstance(other, int):
            assert(isinstance(other, ray.ObjectRef))
            df2 = ray.get(other)
        else:
            df2 = self.cudf_dataframe_dict[other]
        result = func(df1, df2, **kwargs)
        return self.store_new_df(result)

    # test commit
    def reduce(self, first, others, func, axis=0, **kwargs):
        join_func = cudf.DataFrame.join if not axis else lambda x, y: cudf.concat([x,y])
        if not isinstance(others[0], int):
            other_dfs = ray.get(others)
        else:
            other_dfs = [self.cudf_dataframe_dict[i] for i in others]
        df1 = self.cudf_dataframe_dict[first]
        df2 = oids[0] if len(oids) >= 1 else None
        for i in range(1, len(oids)):
            df2 = join_func(df2, other_dfs[i])
        result = func(df1, df2, **kwargs)
        return self.store_new_df(result)

    # store a new key-value pair in cudf_dataframe_dict.
    # key is iterated and then used, so it is key, df.
    # key is returned
    def store_new_df(self, df):
        self.key += 1
        self.cudf_dataframe_dict[self.key] = df
        return self.key

    # if key is in cudf_dataframe_dict, get rid of the key val pair
    def free(self, key):
        if key in self.cudf_dataframe_dict:
            del self.cudf_dataframe_dict[key]

    # getter methods for gpu_id
    def get_id(self):
        return self.gpu_id
    
    # is oid the object id?
    # this assumes dataframe_dict holds oids, so it's key -> oid
    def get_oid(self, key):
        return self.cudf_dataframe_dict[key]

    # take in a pandas df, check if its an instanceof pandas.Series
    # if it is, convert it to a dataFrame with to_frame() (it's like a single column)
    # then add it to the dataframe dict with the pandas_df as the value.
    # so it's a bunch of dataframes?
    def put(self, pandas_df):
        if isinstance(pandas_df, pandas.Series): # if df instanceof a Pandas.series?
            pandas_df = pandas_df.to_frame()
        return self.store_new_df(cudf.from_pandas(pandas_df))
