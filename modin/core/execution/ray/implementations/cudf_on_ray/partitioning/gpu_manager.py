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

"""Module holds Ray actor-class that stores ``cudf.DataFrame``s."""

import ray
import cudf
import pandas


@ray.remote(num_gpus=1)
class GPUManager(object):
    """
    Ray actor-class to store ``cudf.DataFrame``-s and execute functions on it.

    Parameters
    ----------
    gpu_id : int
        The identifier of GPU.
    """

    def __init__(self, gpu_id):
        self.key = 0
        self.cudf_dataframe_dict = {}
        self.gpu_id = gpu_id

    # TODO(#45): Merge apply and apply_non_persistent
    def apply_non_persistent(self, first, other, func, **kwargs):
        """
        Apply `func` to values associated with `first`/`other` keys of `self.cudf_dataframe_dict`.

        Parameters
        ----------
        first : int
            The first key associated with dataframe from `self.cudf_dataframe_dict`.
        other : int
            The second key associated with dataframe from `self.cudf_dataframe_dict`.
            If it isn't a real key, the `func` will be applied to the `first` only.
        func : callable
            A function to apply.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        The type of return of `func`
            The result of the `func` (will be a ``ray.ObjectRef`` in outside level).
        """
        df1 = self.cudf_dataframe_dict[first]
        df2 = self.cudf_dataframe_dict[other] if other else None
        if not df2:
            result = func(df1, **kwargs)
        else:
            result = func(df1, df2, **kwargs)
        return result

    def apply(self, first, other, func, **kwargs):
        """
        Apply `func` to values associated with `first`/`other` keys of `self.cudf_dataframe_dict` with storing of the result.

        Store the return value of `func` (a new ``cudf.DataFrame``)
        into `self.cudf_dataframe_dict`.

        Parameters
        ----------
        first : int
            The first key associated with dataframe from `self.cudf_dataframe_dict`.
        other : int or ray.ObjectRef
            The second key associated with dataframe from `self.cudf_dataframe_dict`.
            If it isn't a real key, the `func` will be applied to the `first` only.
        func : callable
            A function to apply.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        int
            The new key of the new dataframe stored in `self.cudf_dataframe_dict`
            (will be a ``ray.ObjectRef`` in outside level).
        """
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
        """
        Apply `func` to values associated with `first` key and `others` keys of `self.cudf_dataframe_dict` with storing of the result.

        Dataframes associated with `others` keys will be concatenated to one
        dataframe.

        Store the return value of `func` (a new ``cudf.DataFrame``)
        into `self.cudf_dataframe_dict`.

        Parameters
        ----------
        first : int
            The first key associated with dataframe from `self.cudf_dataframe_dict`.
        others : list of int / list of ray.ObjectRef
            The list of keys associated with dataframe from `self.cudf_dataframe_dict`.
        func : callable
            A function to apply.
        axis : {0, 1}, default: 0
            An axis corresponding to a particular row/column of the dataframe.
        **kwargs : dict
            Additional keywords arguments to be passed in `func`.

        Returns
        -------
        int
            The new key of the new dataframe stored in `self.cudf_dataframe_dict`
            (will be a ``ray.ObjectRef`` in outside level).

        Notes
        -----
        If ``len(others) == 0`` `func` should be able to work with 2nd
        positional argument with None value.
        """
        # TODO: Try to use `axis` parameter of cudf.concat
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
        """
        Store `df` in `self.cudf_dataframe_dict`.

        Parameters
        ----------
        df : cudf.DataFrame
            The ``cudf.DataFrame`` to be added.

        Returns
        -------
        int
            The key associated with added dataframe
            (will be a ``ray.ObjectRef`` in outside level).
        """
        self.key += 1
        self.cudf_dataframe_dict[self.key] = df
        return self.key

    def free(self, key):
        """
        Free the dataFrame and associated `key` out of `self.cudf_dataframe_dict`.

        Parameters
        ----------
        key : int
            The key to be deleted.
        """
        if key in self.cudf_dataframe_dict:
            del self.cudf_dataframe_dict[key]

    def get_id(self):
        """
        Get the `self.gpu_id` from this object.

        Returns
        -------
        int
            The gpu_id from this object
            (will be a ``ray.ObjectRef`` in outside level).
        """
        return self.gpu_id

    def get_oid(self, key):
        """
        Get the value from `self.cudf_dataframe_dict` by `key`.

        Parameters
        ----------
        key : int
            The key to get value.

        Returns
        -------
        cudf.DataFrame
            Dataframe corresponding to `key`(will be a ``ray.ObjectRef``
            in outside level).
        """
        return self.cudf_dataframe_dict[key]

    def put(self, pandas_df):
        """
        Convert `pandas_df` to ``cudf.DataFrame`` and put it to `self.cudf_dataframe_dict`.

        Parameters
        ----------
        pandas_df : pandas.DataFrame/pandas.Series
            A pandas DataFrame/Series to be added.

        Returns
        -------
        int
            The key associated with added dataframe
            (will be a ``ray.ObjectRef`` in outside level).
        """
        if isinstance(pandas_df, pandas.Series):
            pandas_df = pandas_df.to_frame()
        return self.store_new_df(cudf.from_pandas(pandas_df))
