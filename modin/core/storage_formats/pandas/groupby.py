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

"""Contains implementations for GroupbyReduce functions."""

import pandas
import numpy as np

from modin.utils import hashable
from modin.core.dataframe.algebra import GroupByReduce
from modin.config import ExperimentalGroupbyImpl
from modin.error_message import ErrorMessage


class GroupbyReduceImpl:
    """Provide TreeReduce implementations for certain groupby aggregations."""

    @classmethod
    def get_impl(cls, agg_name):
        """
        Get TreeReduce implementations for the specified `agg_name`.

        Parameters
        ----------
        agg_name : hashable

        Returns
        -------
        (map_fn: Union[callable, str], reduce_fn: Union[callable, str], default2pandas_fn: callable)
        """
        try:
            return cls._groupby_reduce_impls[agg_name]
        except KeyError:
            raise KeyError(f"Have no implementation for {agg_name}.")

    @classmethod
    def has_impl_for(cls, agg_func):
        """
        Check whether the class has TreeReduce implementation for the specified `agg_func`.

        Parameters
        ----------
        agg_func : hashable or dict

        Returns
        -------
        bool
        """
        if hashable(agg_func):
            return agg_func in cls._groupby_reduce_impls
        if not isinstance(agg_func, dict):
            return False

        # We have to keep this import away from the module level to avoid circular import
        from modin.pandas.utils import walk_aggregation_dict

        for _, func, _, _ in walk_aggregation_dict(agg_func):
            if func not in cls._groupby_reduce_impls:
                return False

        return True

    @classmethod
    def build_qc_method(cls, agg_name, finalizer_fn=None):
        """
        Build a TreeReduce implemented query compiler method for the specified groupby aggregation.

        Parameters
        ----------
        agg_name : hashable
        finalizer_fn : callable(pandas.DataFrame) -> pandas.DataFrame, default: None
            A callable to execute at the end a groupby kernel against groupby result.

        Returns
        -------
        callable
            Function that takes query compiler and executes GroupBy aggregation
            with TreeReduce algorithm.
        """
        map_fn, reduce_fn, d2p_fn = cls.get_impl(agg_name)
        map_reduce_method = GroupByReduce.register(
            map_fn, reduce_fn, default_to_pandas_func=d2p_fn, finalizer_fn=finalizer_fn
        )

        def method(query_compiler, *args, **kwargs):
            if ExperimentalGroupbyImpl.get():
                try:
                    if finalizer_fn is not None:
                        raise NotImplementedError(
                            "Reshuffling groupby is not implemented yet when a finalizing function is specified."
                        )
                    return query_compiler._groupby_shuffle(
                        *args, agg_func=agg_name, **kwargs
                    )
                except NotImplementedError as e:
                    ErrorMessage.warn(
                        f"Can't use experimental reshuffling groupby implementation because of: {e}"
                        + "\nFalling back to a TreeReduce implementation."
                    )
            return map_reduce_method(query_compiler, *args, **kwargs)

        return method

    @staticmethod
    def _build_skew_impl():
        """
        Build TreeReduce implementation for 'skew' groupby aggregation.

        Returns
        -------
        (map_fn: callable, reduce_fn: callable, default2pandas_fn: callable)
        """

        def skew_map(dfgb, *args, **kwargs):
            if dfgb._selection is not None:
                data_to_agg = dfgb._selected_obj
            else:
                cols_to_agg = dfgb.obj.columns.difference(dfgb.exclusions)
                data_to_agg = dfgb.obj[cols_to_agg]

            df_pow2 = data_to_agg**2
            df_pow3 = data_to_agg**3

            return pandas.concat(
                [
                    dfgb.count(*args, **kwargs),
                    dfgb.sum(*args, **kwargs),
                    df_pow2.groupby(dfgb.grouper).sum(*args, **kwargs),
                    df_pow3.groupby(dfgb.grouper).sum(*args, **kwargs),
                ],
                copy=False,
                axis=1,
                keys=["count", "sum", "pow2_sum", "pow3_sum"],
                names=[GroupByReduce.ID_LEVEL_NAME],
            )

        def skew_reduce(dfgb, *args, **kwargs):
            df = dfgb.sum(*args, **kwargs)
            if df.empty:
                return df.droplevel(GroupByReduce.ID_LEVEL_NAME, axis=1)

            count = df["count"]
            s = df["sum"]
            s2 = df["pow2_sum"]
            s3 = df["pow3_sum"]

            # mean = sum(x) / count
            m = s / count

            # m2 = sum( (x - m)^ 2) = sum(x^2 - 2*x*m + m^2)
            m2 = s2 - 2 * m * s + count * (m**2)

            # m3 = sum( (x - m)^ 3) = sum(x^3 - 3*x^2*m + 3*x*m^2 - m^3)
            m3 = s3 - 3 * m * s2 + 3 * s * (m**2) - count * (m**3)

            # The equation for the 'skew' was taken directly from pandas:
            # https://github.com/pandas-dev/pandas/blob/8dab54d6573f7186ff0c3b6364d5e4dd635ff3e7/pandas/core/nanops.py#L1226
            with np.errstate(invalid="ignore", divide="ignore"):
                skew_res = (count * (count - 1) ** 0.5 / (count - 2)) * (m3 / m2**1.5)

            # Setting dummy values for invalid results in accordance with pandas
            skew_res[m2 == 0] = 0
            skew_res[count < 3] = np.nan
            return skew_res

        GroupByReduce.register_implementation(skew_map, skew_reduce)
        return (
            skew_map,
            skew_reduce,
            lambda grp, *args, **kwargs: grp.skew(*args, **kwargs),
        )

    @staticmethod
    def _build_mean_impl():
        """
        Build TreeReduce implementation for 'mean' groupby aggregation.

        Returns
        -------
        (map_fn: callable, reduce_fn: callable, default2pandas_fn: callable)
        """

        def mean_map(dfgb, **kwargs):
            return pandas.concat(
                [dfgb.sum(**kwargs), dfgb.count()],
                axis=1,
                copy=False,
                keys=["sum", "count"],
                names=[GroupByReduce.ID_LEVEL_NAME],
            )

        def mean_reduce(dfgb, **kwargs):
            """
            Compute mean value in each group using sums/counts values within reduce phase.

            Parameters
            ----------
            dfgb : pandas.DataFrameGroupBy
                GroupBy object for column-partition.
            **kwargs : dict
                Additional keyword parameters to be passed in ``pandas.DataFrameGroupBy.sum``.

            Returns
            -------
            pandas.DataFrame
                A pandas Dataframe with mean values in each column of each group.
            """
            sums_counts_df = dfgb.sum(**kwargs)
            if sums_counts_df.empty:
                return sums_counts_df.droplevel(GroupByReduce.ID_LEVEL_NAME, axis=1)

            sum_df = sums_counts_df["sum"]
            count_df = sums_counts_df["count"]

            return sum_df / count_df

        GroupByReduce.register_implementation(mean_map, mean_reduce)

        return (
            mean_map,
            mean_reduce,
            lambda grp, *args, **kwargs: grp.mean(*args, **kwargs),
        )


GroupbyReduceImpl._groupby_reduce_impls = {
    "all": ("all", "all", lambda grp, *args, **kwargs: grp.all(*args, **kwargs)),
    "any": ("any", "any", lambda grp, *args, **kwargs: grp.any(*args, **kwargs)),
    "count": ("count", "sum", lambda grp, *args, **kwargs: grp.count(*args, **kwargs)),
    "max": ("max", "max", lambda grp, *args, **kwargs: grp.max(*args, **kwargs)),
    "mean": GroupbyReduceImpl._build_mean_impl(),
    "min": ("min", "min", lambda grp, *args, **kwargs: grp.min(*args, **kwargs)),
    "prod": ("prod", "prod", lambda grp, *args, **kwargs: grp.prod(*args, **kwargs)),
    "size": ("size", "sum", lambda grp, *args, **kwargs: grp.size(*args, **kwargs)),
    "skew": GroupbyReduceImpl._build_skew_impl(),
    "sum": ("sum", "sum", lambda grp, *args, **kwargs: grp.sum(*args, **kwargs)),
}
