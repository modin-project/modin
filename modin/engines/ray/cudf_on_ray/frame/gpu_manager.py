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
from .partition import cuDFOnRayFramePartition ## imported from partition.py.


# ray remote -- parallelizes a class/function
@ray.remote(num_gpus=1) 
class GPUManager(object):

    # constructor/initializer method, takes gpu_id.
    def __init__(self, gpu_id):
        self.key = 0
        self.cudf_dataframe_dict = {}  # holds cudf DataFrames or cudf Series.
        self.gpu_id = gpu_id

    def apply_non_persistent(self, first, other, func, **kwargs):
        """ 
        Given two keys, apply a function using the values associated with the keys as params.
        Return the value of the function.

        Parameters
        ---------
            first : int
                The first key. You will get a dataframe out of the dataframe_dict with this, store it into df1.
            other : int
                The second key. If it isn't a real key, then it's a none. In such a case, func is called with two params
                instead of three.
            func : func
                A function that we will use/apply on the two other params (first, other).
            **kwargs: dict
                An iterable object that corresponds to a dict, if i'm not mistaken.  
        Returns
        -------
            result
                the result of the function (will be an OID).
        """
        df1 = self.cudf_dataframe_dict[first]
        df2 = self.cudf_dataframe_dict[other] if other else None # if other is not passed in, None.
        if not df2:
            result = func(df1, **kwargs)
        else:
            result = func(df1, df2, **kwargs)
        return result

    def apply(self, first, other, func, **kwargs):
        """ 
        Given two keys, apply a function using the dataFrames from
        the cudf_dataframe_dict associated with the keys.
        Store the return value of the function (a new cudf_DataFrame)
        into cudf_dataframe_dict. Return the new key associated with this value 
        (will be an OID).

        Parameters
        ---------
            first : int
                The first key. You will get a dataframe out of the dataframe_dict with this, store it into df1.
            other : int
                The second key. If it isn't a real key, then it's an objectRef, and we must get the actual dataFrame
                with ray.get(other).
                instead of three.
            func : func
                A function that we will use/apply on the two other params (first, other).
            **kwargs: dict
                An iterable object that corresponds to a dict, if i'm not mistaken.  
        Returns
        -------
            self.store_new_df(result) : int
                the new key of the new dataFrame stored in cudf_dataframe_dict.
        """
        df1 = self.cudf_dataframe_dict[first]
        if not isinstance(other, int):
            assert(isinstance(other, ray.ObjectRef))
            df2 = ray.get(other)
        else:
            df2 = self.cudf_dataframe_dict[other]
        result = func(df1, df2, **kwargs)
        return self.store_new_df(result)


    # reduce
    # we join via cudf.DataFrame join if the axis isn't real/doesn't exist,
    # else, we run a lambda expression, where we take in params x,y, and concat them.
    # if others[0] isnt an integer, we assume it's an object id, and we get the actual
    # information. Else, we do list comprehension. We create a list of items/dataframes/oids.
    # DF1 = the value of indexing into dataframe_dict with first.
    # DF2 = the first value of oids/other_dfs (it has to be fixed).
    # we iterate through the length of the others array, joining these items together.
    # then we run a function on df1, drf2, and we store the result into the dataframe_dict
    # and return the key.
    def reduce(self, first, others, func, axis=0, **kwargs):
        """ 
        Given two keys, apply a function using the dataFrames from
        the cudf_dataframe_dict associated with the keys.
        Store the return value of the function (a new cudf_DataFrame)
        into cudf_dataframe_dict. Return the new key associated with this value 
        (will be an OID).

        Parameters
        ---------
            first : int
                The first key. You will get a dataframe out of the dataframe_dict with this, store it into df1.
            other : int
                The second key. If it isn't a real key, then it's an objectRef, and we must get the actual dataFrame
                with ray.get(other).
                instead of three.
            func : func
                A function that we will use/apply on the two other params (first, other).
            axis : ? (what type)
                An axis corresponding to a particular row/column of the dataFrame.
            **kwargs: dict
                An iterable object that corresponds to a dict, if i'm not mistaken.  
        Returns
        -------
            self.store_new_df(result) : int
                the new key of the new dataFrame stored in cudf_dataframe_dict.
        """
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

    def store_new_df(self, df):
        """
        Store a new cudf_dataFrame in the dataframe_dict.
        Iterate the current key int, and store the dataFrame in the dict
        with this new iterated key
        Return the key associated with this new cudf_dataFrame.
        
        Will be an OID corresponding to an int key.        

        Parameters 
        ----------
            df : dataFrame
                This is a dataFrame we're adding to cudf_dataframe_dict.
        
        Returns 
        ------
            self.key : int
                This is the key associated the value dataFrame we passed in.

        """
        self.key += 1
        self.cudf_dataframe_dict[self.key] = df
        return self.key

    
    def free(self, key):
        """
        Free the dataFrame and associated key out of the dataframe_dict.

        Parameters
        ----------
            key : int
                The key we want to free (deletes the key-val pair).
        """
        if key in self.cudf_dataframe_dict:
            del self.cudf_dataframe_dict[key]

   
    def get_id(self):
        """
        Get the gpu_id from the gpu_manager object.

        Returns
        -------
            self.gpu_id : int
                the gpu_id from the gpu_manager object.
                It will be represented as an OID, naturally.
        """
        return self.gpu_id
   
    def get_oid(self, key):         
        """
        Given a key, return the value of the key-val pair from cudf_dataframe_dict. 
        
        Parameters
        ----------
            key : int
                The integer that corresponds to a particular key-value pair.
        
        Returns
        -------
            an oid corresponding to a cudf dataframe from cudf_dataframe_dict 
            (wrapped up as an OID).
        """
        return self.cudf_dataframe_dict[key]

    

    def put(self, pandas_df):
        """
        Given a pandas_df object, 
        convert it to a cudf_DataFrame, and add it to the cudf_dataframe_dict. 
        Return the new key added to the dictionary (as an OID).
        
        Parameters
        ----------
            pandas_df : pandas_df
                the pandas dataFrame object.
                It may or may not be a pandas.Series object. 
                If it is, convert it to a dataFrame.
                Then, convert it to a cudf.dataFrame.

        Returns
        -------
            an oid corresponding to the key generated 
            when you added the new cudf_DataFrame object to the cudf_dataframe_dict.
        """
        if isinstance(pandas_df, pandas.Series): # if df instanceof a Pandas.series?
            pandas_df = pandas_df.to_frame()
        return self.store_new_df(cudf.from_pandas(pandas_df))
