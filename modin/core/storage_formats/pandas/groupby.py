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

import numpy as np
import pandas
from pandas.core.dtypes.common import is_list_like

from modin.config import RangePartitioningGroupby
from modin.core.dataframe.algebra import GroupByReduce
from modin.error_message import ErrorMessage
from modin.utils import hashable


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
            if RangePartitioningGroupby.get():
                try:
                    if finalizer_fn is not None:
                        raise NotImplementedError(
                            "Range-partitioning groupby is not implemented yet when a finalizing function is specified."
                        )
                    return query_compiler._groupby_shuffle(
                        *args, agg_func=agg_name, **kwargs
                    )
                except NotImplementedError as e:
                    ErrorMessage.warn(
                        f"Can't use range-partitioning groupby implementation because of: {e}"
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


class PivotTableImpl:
    """Provide MapReduce, Range-Partitioning and Full-Column implementations for 'pivot_table()'."""

    @classmethod
    def map_reduce_impl(
        cls,
        qc,
        index,
        values,
        columns,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
        sort,
    ):  # noqa: PR01
        """Compute 'pivot_table()' using MapReduce implementation."""
        if margins:
            raise NotImplementedError(
                "MapReduce 'pivot_table' implementation doesn't support 'margins=True' parameter"
            )

        if not GroupbyReduceImpl.has_impl_for(aggfunc):
            raise NotImplementedError(
                "MapReduce 'pivot_table' implementation only supports 'aggfuncs' that are implemented in 'GroupbyReduceImpl'"
            )

        index, columns, values, drop_column_level, unique_keys = cls._preprocess_args(
            index, columns, values
        )

        if len(set(index).intersection(columns)) > 0:
            raise NotImplementedError(
                "MapReduce 'pivot_table' implementation doesn't support intersections of 'index' and 'columns'"
            )

        to_group, keys_columns = cls._separate_data_from_grouper(
            qc, values, unique_keys
        )
        to_unstack = columns if index else None

        result = GroupbyReduceImpl.build_qc_method(
            aggfunc,
            finalizer_fn=lambda df: cls._pivot_table_from_groupby(
                df, dropna, drop_column_level, to_unstack, fill_value
            ),
        )(
            to_group,
            by=keys_columns,
            axis=0,
            groupby_kwargs={"observed": observed, "sort": sort},
            agg_args=(),
            agg_kwargs={},
            drop=True,
        )

        if to_unstack is None:
            result = result.transpose()
        return result

    @classmethod
    def full_axis_impl(
        cls,
        qc,
        index,
        values,
        columns,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
        sort,
    ):  # noqa: PR01
        """Compute 'pivot_table()' using full-column-axis implementation."""
        index, columns, values, _, unique_keys = cls._preprocess_args(
            index, columns, values
        )
        to_group, keys_columns = cls._separate_data_from_grouper(
            qc, values, unique_keys
        )

        len_values = len(values)
        if len_values == 0:
            len_values = len(qc.columns.drop(unique_keys))

        def applyier(df, other):  # pragma: no cover
            """
            Build pivot table for a single partition.

            Parameters
            ----------
            df : pandas.DataFrame
                Partition of the self frame.
            other : pandas.DataFrame
                Broadcasted partition that contains `value` columns
                of the self frame.

            Returns
            -------
            pandas.DataFrame
                Pivot table for this particular partition.
            """
            concated = pandas.concat([df, other], axis=1, copy=False)
            result = pandas.pivot_table(
                concated,
                index=index,
                values=values if len(values) > 0 else None,
                columns=columns,
                aggfunc=aggfunc,
                fill_value=fill_value,
                margins=margins,
                dropna=dropna,
                margins_name=margins_name,
                observed=observed,
                sort=sort,
            )

            # if only one value is specified, removing level that maps
            # columns from `values` to the actual values
            if len(index) > 0 and len_values == 1 and result.columns.nlevels > 1:
                result.columns = result.columns.droplevel(int(margins))

            # in that case Pandas transposes the result of `pivot_table`,
            # transposing it back to be consistent with column axis values along
            # different partitions
            if len(index) == 0 and len(columns) > 0:
                result = result.T

            return result

        result = qc.__constructor__(
            to_group._modin_frame.broadcast_apply_full_axis(
                axis=0, func=applyier, other=keys_columns._modin_frame
            )
        )

        # transposing the result again, to be consistent with Pandas result
        if len(index) == 0 and len(columns) > 0:
            result = result.transpose()

        return result

    @classmethod
    def range_partition_impl(
        cls,
        qc,
        index,
        values,
        columns,
        aggfunc,
        fill_value,
        margins,
        dropna,
        margins_name,
        observed,
        sort,
    ):  # noqa: PR01
        """Compute 'pivot_table()' using Range-Partitioning implementation."""
        if margins:
            raise NotImplementedError(
                "Range-partitioning 'pivot_table' implementation doesn't support 'margins=True' parameter"
            )

        index, columns, values, drop_column_level, unique_keys = cls._preprocess_args(
            index, columns, values
        )

        if len(values) > 0:
            to_take = list(np.unique(list(index) + list(columns) + list(values)))
            qc = qc.getitem_column_array(to_take, ignore_order=True)

        if len(set(index).intersection(columns)) > 0:
            raise NotImplementedError(
                "Range-partitioning 'pivot_table' implementation doesn't support intersections of 'index' and 'columns'"
            )

        to_unstack = columns if index else None

        groupby_result = qc._groupby_shuffle(
            by=list(unique_keys),
            agg_func=aggfunc,
            axis=0,
            groupby_kwargs={"observed": observed, "sort": sort},
            agg_args=(),
            agg_kwargs={},
            drop=True,
        )

        result = qc.__constructor__(
            groupby_result._modin_frame.apply_full_axis(
                axis=0,
                func=lambda df: cls._pivot_table_from_groupby(
                    df, dropna, drop_column_level, to_unstack, fill_value
                ),
            )
        )

        if to_unstack is None:
            result = result.transpose()

        return result

    @staticmethod
    def _pivot_table_from_groupby(
        df, dropna, drop_column_level, to_unstack, fill_value
    ):
        """
        Convert group by aggregation result to a pivot table.

        Parameters
        ----------
        df : pandas.DataFrame
            Group by aggregation result.
        dropna : bool
            Whether to drop NaN columns.
        drop_column_level : bool
            Whether to drop the top-columns level.
        to_unstack : list of labels or None
            Group by keys to pass to ``.unstack()``. Reperent `columns` parameter
            for ``.pivot_table()``.
        fill_value : bool
            Fill value for NaN values.

        Returns
        -------
        pandas.DataFrame
        """
        if df.index.nlevels > 1 and to_unstack is not None:
            df = df.unstack(level=to_unstack)
        if drop_column_level and df.columns.nlevels > 1:
            df = df.droplevel(0, axis=1)
        if dropna:
            df = df.dropna(axis=1, how="all")
        if fill_value is not None:
            df = df.fillna(fill_value, downcast="infer")
        return df

    @staticmethod
    def _preprocess_args(index, columns, values):
        """
        Normalize `.pivot_table()` args by converting them to a list.

        Parameters
        ----------
        index : list, scalar or None
        columns : list, scalar or None
        values : list, scalar or None

        Returns
        -------
        index : list
        columns : list
        values : list
        drop_column_level : bool
            Whether to drop the top-columns level after group by aggregation.
        unique_keys : list of labels
            Columns to group by.
        """
        from pandas.core.reshape.pivot import _convert_by

        def __convert_by(by):
            """Convert passed value to a list."""
            if isinstance(by, pandas.Index):
                return list(by)
            return _convert_by(by)

        drop_column_level = values is not None and not is_list_like(values)
        index, columns, values = map(__convert_by, [index, columns, values])
        # using 'pandas.unique' instead of 'numpy' as it guarantees to not change the original order
        unique_keys = pandas.unique(index + columns)

        return index, columns, values, drop_column_level, unique_keys

    @staticmethod
    def _separate_data_from_grouper(qc, values, unique_keys):
        """
        Split `qc` for key columns to group by and values to aggregate.

        Parameters
        ----------
        qc : PandasQueryCompiler
        values : list of labels
            List of columns to aggregate.
        unique_keys : list of labels
            List of key columns to group by.

        Returns
        -------
        to_aggregate : PandasQueryCompiler
        keys_to_group : PandasQueryCompiler
        """
        unique_values = np.unique(values)

        if len(values):
            to_aggregate = qc.getitem_column_array(unique_values, ignore_order=True)
        else:
            to_aggregate = qc.drop(columns=unique_keys)

        keys_to_group = qc.getitem_column_array(unique_keys, ignore_order=True)

        return to_aggregate, keys_to_group
