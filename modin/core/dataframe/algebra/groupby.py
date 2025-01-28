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

"""Module houses builder class for GroupByReduce operator."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, Union

import pandas

from modin.core.dataframe.pandas.metadata import ModinIndex
from modin.error_message import ErrorMessage
from modin.utils import MODIN_UNNAMED_SERIES_LABEL, hashable

from .default2pandas.groupby import GroupBy, GroupByDefault
from .tree_reduce import TreeReduce

if TYPE_CHECKING:
    from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler


class GroupByReduce(TreeReduce):
    """
    Builder class for GroupBy aggregation functions.

    Attributes
    ----------
    ID_LEVEL_NAME : str
        It's supposed that implementations may produce multiple temporary
        columns per one source column in an intermediate phase. In order
        for these columns to be processed accordingly at the Reduce phase,
        an implementation must store unique names for such temporary
        columns in the ``ID_LEVEL_NAME`` level. Duplicated names are not allowed.
    _GROUPBY_REDUCE_IMPL_FLAG : str
        Attribute indicating that a callable should be treated as an
        implementation for one of the TreeReduce phases rather than an
        arbitrary aggregation. Note: this attribute should be considered private.
    """

    ID_LEVEL_NAME: str = "__ID_LEVEL_NAME__"
    _GROUPBY_REDUCE_IMPL_FLAG: str = "__groupby_reduce_impl_func__"

    @classmethod
    def register(
        cls,
        map_func: Union[str, dict, Callable[..., pandas.DataFrame]],
        reduce_func: Optional[Union[str, dict, Callable[..., pandas.DataFrame]]] = None,
        **call_kwds: dict,
    ) -> Callable[..., PandasQueryCompiler]:
        """
        Build template GroupBy aggregation function.

        Resulted function is applied in parallel via TreeReduce algorithm.

        Parameters
        ----------
        map_func : str, dict or callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the map phase. If ``str`` was passed it will
            be treated as a DataFrameGroupBy's method name.
        reduce_func : str, dict or callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame, optional
            Function to apply to the ``DataFrameGroupBy`` at the reduce phase. If not specified
            will be set the same as 'map_func'.
        **call_kwds : dict
            Kwargs that will be passed to the returned function.

        Returns
        -------
        callable
            Function that takes query compiler and executes GroupBy aggregation
            with TreeReduce algorithm.
        """
        if reduce_func is None:
            reduce_func = map_func

        def build_fn(name):
            return lambda df, *args, **kwargs: getattr(df, name)(*args, **kwargs)

        if isinstance(map_func, str):
            map_func = build_fn(map_func)
        if isinstance(reduce_func, str):
            reduce_func = build_fn(reduce_func)

        assert not (
            isinstance(map_func, dict) ^ isinstance(reduce_func, dict)
        ) and not (
            callable(map_func) ^ callable(reduce_func)
        ), "Map and reduce functions must be either both dict or both callable."

        return lambda *args, **kwargs: cls.caller(
            *args, map_func=map_func, reduce_func=reduce_func, **kwargs, **call_kwds
        )

    @classmethod
    def register_implementation(
        cls,
        map_func: Callable[..., pandas.DataFrame],
        reduce_func: Callable[..., pandas.DataFrame],
    ) -> None:
        """
        Register callables to be recognized as an implementations of tree-reduce phases.

        Parameters
        ----------
        map_func : callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
            Callable to register.
        reduce_func : callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
            Callable to register.
        """
        setattr(map_func, cls._GROUPBY_REDUCE_IMPL_FLAG, True)
        setattr(reduce_func, cls._GROUPBY_REDUCE_IMPL_FLAG, True)

    @classmethod
    def map(
        cls,
        df: pandas.DataFrame,
        map_func: Callable[..., pandas.DataFrame],
        axis: int,
        groupby_kwargs: dict,
        agg_args: list,
        agg_kwargs: dict,
        other: Optional[pandas.DataFrame] = None,
        by=None,
        drop: bool = False,
    ) -> pandas.DataFrame:
        """
        Execute Map phase of GroupByReduce.

        Groups DataFrame and applies map function. Groups will be
        preserved in the results index for the following reduce phase.

        Parameters
        ----------
        df : pandas.DataFrame
            Serialized frame to group.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject`.
        axis : {0, 1}
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_kwargs : dict
            Dictionary which carries arguments for `pandas.DataFrame.groupby`.
        agg_args : list-like
            Positional arguments to pass to the aggregation functions.
        agg_kwargs : dict
            Keyword arguments to pass to the aggregation functions.
        other : pandas.DataFrame, optional
            Serialized frame, whose columns are used to determine the groups.
            If not specified, `by` parameter is used.
        by : level index name or list of such labels, optional
            Index levels, that is used to determine groups.
            If not specified, `other` parameter is used.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.

        Returns
        -------
        pandas.DataFrame
            GroupBy aggregation result for one particular partition.
        """
        # Set `as_index` to True to track the metadata of the grouping object
        # It is used to make sure that between phases we are constructing the
        # right index and placing columns in the correct order.
        groupby_kwargs["as_index"] = True
        groupby_kwargs["observed"] = True
        # We have to filter func-dict BEFORE inserting broadcasted 'by' columns
        # to avoid multiple aggregation results for 'by' cols in case they're
        # present in the func-dict:
        apply_func = cls.get_callable(
            map_func,
            df,
            # We won't be able to preserve the order as the Map phase would likely
            # produce some temporary columns that won't fit into the original
            # aggregation order. It doesn't matter much as we restore the original
            # order at the Reduce phase.
            preserve_aggregation_order=False,
        )
        if other is not None:
            # Other is a broadcasted partition that represents 'by' data to group on.
            # If 'drop' then the 'by' data came from the 'self' frame, thus
            # inserting missed columns to the partition to group on them.
            if drop or isinstance(
                other := other.squeeze(axis=axis ^ 1), pandas.DataFrame
            ):
                df = pandas.concat(
                    [df] + [other[[o for o in other if o not in df]]],
                    axis=1,
                )
                other = list(other.columns)
            by_part = other
        else:
            by_part = by

        result = apply_func(
            df.groupby(by=by_part, axis=axis, **groupby_kwargs), *agg_args, **agg_kwargs
        )
        # Result could not always be a frame, so wrapping it into DataFrame
        return pandas.DataFrame(result)

    @classmethod
    def reduce(
        cls,
        df: pandas.DataFrame,
        reduce_func: Union[dict, Callable[..., pandas.DataFrame]],
        axis: int,
        groupby_kwargs: dict,
        agg_args: list,
        agg_kwargs: dict,
        partition_idx: int = 0,
        drop: bool = False,
        method: Optional[str] = None,
        finalizer_fn: Optional[Callable[[pandas.DataFrame], pandas.DataFrame]] = None,
    ) -> pandas.DataFrame:
        """
        Execute Reduce phase of GroupByReduce.

        Combines groups from the Map phase and applies reduce function.

        Parameters
        ----------
        df : pandas.DataFrame
            Serialized frame which contain groups to combine.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject`.
        axis : {0, 1}
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_kwargs : dict
            Dictionary which carries arguments for `pandas.DataFrame.groupby`.
        agg_args : list-like
            Positional arguments to pass to the aggregation functions.
        agg_kwargs : dict
            Keyword arguments to pass to the aggregation functions.
        partition_idx : int, default: 0
            Internal index of column partition to which this function is applied.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
        method : str, optional
            Name of the groupby function. This is a hint to be able to do special casing.
        finalizer_fn : callable(pandas.DataFrame) -> pandas.DataFrame, optional
            A callable to execute at the end a groupby kernel against groupby result.

        Returns
        -------
        pandas.DataFrame
            GroupBy aggregation result.
        """
        # Wrapping names into an Index should be unnecessary, however
        # there is a bug in pandas with intersection that forces us to do so:
        # https://github.com/pandas-dev/pandas/issues/39699
        by_part = pandas.Index(df.index.names)

        groupby_kwargs = groupby_kwargs.copy()
        as_index = groupby_kwargs.get("as_index", True)

        # Set `as_index` to True to track the metadata of the grouping object
        groupby_kwargs["as_index"] = True

        # since now index levels contain out 'by', in the reduce phace
        # we want to group on these levels
        groupby_kwargs["level"] = list(range(len(df.index.names)))

        apply_func = cls.get_callable(reduce_func, df)
        result = apply_func(
            df.groupby(axis=axis, **groupby_kwargs), *agg_args, **agg_kwargs
        )

        if not as_index:
            idx = df.index
            GroupBy.handle_as_index_for_dataframe(
                result,
                by_part,
                by_cols_dtypes=(
                    idx.dtypes.values
                    if isinstance(idx, pandas.MultiIndex) and hasattr(idx, "dtypes")
                    else (idx.dtype,)
                ),
                by_length=len(by_part),
                selection=reduce_func.keys() if isinstance(reduce_func, dict) else None,
                partition_idx=partition_idx,
                drop=drop,
                method=method,
                inplace=True,
            )
        # Result could not always be a frame, so wrapping it into DataFrame
        result = pandas.DataFrame(result)
        if result.index.name == MODIN_UNNAMED_SERIES_LABEL:
            result.index.name = None

        return result if finalizer_fn is None else finalizer_fn(result)

    @classmethod
    def caller(
        cls,
        query_compiler: PandasQueryCompiler,
        by,
        map_func: Union[dict, Callable[..., pandas.DataFrame]],
        reduce_func: Union[dict, Callable[..., pandas.DataFrame]],
        axis: int,
        groupby_kwargs: dict,
        agg_args: list,
        agg_kwargs: dict,
        drop: bool = False,
        method: Optional[str] = None,
        default_to_pandas_func: Optional[Callable[..., pandas.DataFrame]] = None,
        finalizer_fn: Optional[Callable[[pandas.DataFrame], pandas.DataFrame]] = None,
    ) -> PandasQueryCompiler:
        """
        Execute GroupBy aggregation with TreeReduce approach.

        Parameters
        ----------
        query_compiler : PandasQueryCompiler
            Frame to group.
        by : PandasQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Map phase.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Reduce phase.
        axis : {0, 1}
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_kwargs : dict
            Dictionary which carries arguments for pandas.DataFrame.groupby.
        agg_args : list-like
            Positional arguments to pass to the aggregation functions.
        agg_kwargs : dict
            Keyword arguments to pass to the aggregation functions.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
        method : str, optional
            Name of the GroupBy aggregation function. This is a hint to be able to do special casing.
        default_to_pandas_func : callable(pandas.DataFrameGroupBy) -> pandas.DataFrame, optional
            The pandas aggregation function equivalent to the `map_func + reduce_func`.
            Used in case of defaulting to pandas. If not specified `map_func` is used.
        finalizer_fn : callable(pandas.DataFrame) -> pandas.DataFrame, optional
            A callable to execute at the end a groupby kernel against groupby result.

        Returns
        -------
        PandasQueryCompiler
            QueryCompiler which carries the result of GroupBy aggregation.
        """
        is_unsupported_axis = axis != 0
        # Defaulting to pandas in case of an empty frame as we can't process it properly.
        # Higher API level won't pass empty data here unless the frame has delayed
        # computations. So we apparently lose some laziness here (due to index access)
        # because of the inability to process empty groupby natively.
        is_empty_data = (
            len(query_compiler.columns) == 0 or len(query_compiler.index) == 0
        )
        is_grouping_using_by_arg = (
            groupby_kwargs.get("level", None) is None and by is not None
        )
        is_unsupported_by_arg = isinstance(by, pandas.Grouper) or (
            not hashable(by) and not isinstance(by, type(query_compiler))
        )

        if (
            is_unsupported_axis
            or is_empty_data
            or (is_grouping_using_by_arg and is_unsupported_by_arg)
        ):
            if default_to_pandas_func is None:
                default_to_pandas_func = (
                    (lambda grp: grp.agg(map_func))
                    if isinstance(map_func, dict)
                    else map_func
                )
            default_to_pandas_func = GroupByDefault.register(default_to_pandas_func)
            return default_to_pandas_func(
                query_compiler,
                by=by,
                axis=axis,
                groupby_kwargs=groupby_kwargs,
                agg_args=agg_args,
                agg_kwargs=agg_kwargs,
                drop=drop,
            )

        # The bug only occurs in the case of Categorical 'by', so we might want to check whether any of
        # the 'by' dtypes is Categorical before going into this branch, however triggering 'dtypes'
        # computation if they're not computed may take time, so we don't do it
        if not groupby_kwargs.get("sort", True) and isinstance(
            by, type(query_compiler)
        ):
            ErrorMessage.mismatch_with_pandas(
                operation="df.groupby(categorical_by, sort=False)",
                message=(
                    "the groupby keys will be sorted anyway, although the 'sort=False' was passed. "
                    + "See the following issue for more details: "
                    + "https://github.com/modin-project/modin/issues/3571"
                ),
            )
            groupby_kwargs = groupby_kwargs.copy()
            groupby_kwargs["sort"] = True

        map_fn, reduce_fn = cls.build_map_reduce_functions(
            by=by,
            axis=axis,
            groupby_kwargs=groupby_kwargs,
            map_func=map_func,
            reduce_func=reduce_func,
            agg_args=agg_args,
            agg_kwargs=agg_kwargs,
            drop=drop,
            method=method,
            finalizer_fn=finalizer_fn,
        )

        # If `by` is a ModinFrame, then its partitions will be broadcasted to every
        # `self` partition in a way determined by engine (modin_frame.groupby_reduce)
        # Otherwise `by` was already bound to the Map function in `build_map_reduce_functions`.
        broadcastable_by = getattr(by, "_modin_frame", None)
        apply_indices = list(map_func.keys()) if isinstance(map_func, dict) else None
        if (
            broadcastable_by is not None
            and groupby_kwargs.get("as_index", True)
            and broadcastable_by.has_materialized_dtypes
        ):
            new_index = ModinIndex(
                # actual value will be assigned on a parent update
                value=None,
                axis=0,
                dtypes=broadcastable_by.dtypes,
            )
        else:
            new_index = None
        new_modin_frame = query_compiler._modin_frame.groupby_reduce(
            axis,
            broadcastable_by,
            map_fn,
            reduce_fn,
            apply_indices=apply_indices,
            new_index=new_index,
        )

        result = query_compiler.__constructor__(new_modin_frame)
        return result

    @classmethod
    def get_callable(
        cls,
        agg_func: Union[dict, Callable[..., pandas.DataFrame]],
        df: pandas.DataFrame,
        preserve_aggregation_order: bool = True,
    ) -> Callable[..., pandas.DataFrame]:
        """
        Build aggregation function to apply to each group at this particular partition.

        If it's dictionary aggregation — filters aggregation dictionary for keys which
        this particular partition contains, otherwise do nothing with passed function.

        Parameters
        ----------
        agg_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Aggregation function.
        df : pandas.DataFrame
            Serialized partition which contains available columns.
        preserve_aggregation_order : bool, default: True
            Whether to manually restore the order of columns for the result specified
            by the `agg_func` keys (only makes sense when `agg_func` is a dictionary).

        Returns
        -------
        callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
            Aggregation function that can be safely applied to this particular partition.
        """
        if not isinstance(agg_func, dict):
            return agg_func

        grp_has_id_level = df.columns.names[0] == cls.ID_LEVEL_NAME
        # The 'id' level prevents us from a lookup for the original
        # partition's columns. So dropping the level.
        partition_columns = frozenset(
            df.columns.droplevel(0) if grp_has_id_level else df.columns
        )

        partition_dict = {k: v for k, v in agg_func.items() if k in partition_columns}
        return cls._build_callable_for_dict(
            partition_dict, preserve_aggregation_order, grp_has_id_level
        )

    @classmethod
    def _build_callable_for_dict(
        cls,
        agg_dict: dict,
        preserve_aggregation_order: bool = True,
        grp_has_id_level: bool = False,
    ) -> Callable[..., pandas.DataFrame]:
        """
        Build callable for an aggregation dictionary.

        Parameters
        ----------
        agg_dict : dict
            Aggregation dictionary.
        preserve_aggregation_order : bool, default: True
            Whether to manually restore the order of columns for the result specified
            by the `agg_func` keys (only makes sense when `agg_func` is a dictionary).
        grp_has_id_level : bool, default: False
            Whether the frame we're grouping on has intermediate columns
            (see ``cls.ID_LEVEL_NAME``).

        Returns
        -------
        callable(pandas.core.groupby.DataFrameGroupBy) -> pandas.DataFrame
        """
        # We have to keep this import away from the module level to avoid circular import
        from modin.pandas.utils import walk_aggregation_dict

        # We now filter aggregation functions into those that could be applied natively
        # using pandas (pandas_grp_obj.agg(**native_aggs)) and those that require
        # special treatment (custom_aggs).
        custom_aggs = {}
        native_aggs = {}

        result_columns = []
        for col, func, func_name, col_renaming_required in walk_aggregation_dict(
            agg_dict
        ):
            # Filter dictionary
            dict_to_add = (
                custom_aggs if cls.is_registered_implementation(func) else native_aggs
            )

            new_value = func if func_name is None else (func_name, func)
            old_value = dict_to_add.get(col, None)

            if old_value is not None:
                ErrorMessage.catch_bugs_and_request_email(
                    failure_condition=not isinstance(old_value, list),
                    extra_log="Expected for all aggregation values to be a list when at least "
                    + f"one column has multiple aggregations. Got: {old_value} {type(old_value)}",
                )
                old_value.append(new_value)
            else:
                # Pandas knows that it has to modify the resulting columns if it meets
                # a function wrapped into a list. Renaming is required if either a new
                # column name was explicitly specified, or multiple functions were
                # specified per one column, or if any other column in the aggregation
                # is going to be renamed.
                dict_to_add[col] = [new_value] if col_renaming_required else new_value

            # Construct resulting columns
            if col_renaming_required:
                func_name = str(func) if func_name is None else func_name
                result_columns.append(
                    (*(col if isinstance(col, tuple) else (col,)), func_name)
                )
            else:
                result_columns.append(col)

        result_columns = pandas.Index(result_columns)

        def aggregate_on_dict(grp_obj, *args, **kwargs):
            """Aggregate the passed groupby object."""
            if len(native_aggs) == 0:
                native_agg_res = None
            elif grp_has_id_level:
                # Adding the 'id' level to the aggregation keys so they match `grp_obj` columns
                native_aggs_modified = {
                    (
                        cls.ID_LEVEL_NAME,
                        *(key if isinstance(key, tuple) else (key,)),
                    ): value
                    for key, value in native_aggs.items()
                }
                native_agg_res = grp_obj.agg(native_aggs_modified)
                # Dropping the 'id' level from the resulted frame
                native_agg_res.columns = native_agg_res.columns.droplevel(0)
            else:
                native_agg_res = grp_obj.agg(native_aggs)

            custom_results = []
            insert_id_levels = False

            for col, func, func_name, col_renaming_required in walk_aggregation_dict(
                custom_aggs
            ):
                if grp_has_id_level:
                    cols_without_ids = grp_obj.obj.columns.droplevel(0)
                    if isinstance(cols_without_ids, pandas.MultiIndex):
                        # We may have multiple columns matching the `col` in
                        # a MultiIndex case, that's why use `.get_locs` here
                        col_pos = cols_without_ids.get_locs(col)
                    else:
                        # `pandas.Index` doesn't have `.get_locs` method
                        col_pos = cols_without_ids.get_loc(col)
                    agg_key = grp_obj.obj.columns[col_pos]
                else:
                    agg_key = [col]

                result = func(grp_obj[agg_key])
                # The `func` may have discarded an ID-level if there were any.
                # So checking for this again.
                result_has_id_level = result.columns.names[0] == cls.ID_LEVEL_NAME
                insert_id_levels |= result_has_id_level

                if col_renaming_required:
                    func_name = str(func) if func_name is None else func_name
                    if result_has_id_level:
                        result.columns = pandas.MultiIndex.from_tuples(
                            [
                                # `old_col[0]` stores values from the 'id'
                                # level, the ones we want to preserve here
                                (old_col[0], col, func_name)
                                for old_col in result.columns
                            ],
                            names=[
                                result.columns.names[0],
                                result.columns.names[1],
                                None,
                            ],
                        )
                    else:
                        result.columns = pandas.MultiIndex.from_tuples(
                            [(col, func_name)] * len(result.columns),
                            names=[result.columns.names[0], None],
                        )

                custom_results.append(result)

            if insert_id_levels:
                # As long as any `result` has an id-level we have to insert the level
                # into every `result` so the number of levels matches
                for idx, ext_result in enumerate(custom_results):
                    if ext_result.columns.names[0] != cls.ID_LEVEL_NAME:
                        custom_results[idx] = pandas.concat(
                            [ext_result],
                            keys=[cls.ID_LEVEL_NAME],
                            names=[cls.ID_LEVEL_NAME],
                            axis=1,
                            copy=False,
                        )

                if native_agg_res is not None:
                    native_agg_res = pandas.concat(
                        [native_agg_res],
                        keys=[cls.ID_LEVEL_NAME],
                        names=[cls.ID_LEVEL_NAME],
                        axis=1,
                        copy=False,
                    )

            native_res_part = [] if native_agg_res is None else [native_agg_res]
            parts = [*native_res_part, *custom_results]
            if parts:
                result = pandas.concat(parts, axis=1, copy=False)
            else:
                result = pandas.DataFrame(columns=result_columns)

            # The order is naturally preserved if there's no custom aggregations
            if preserve_aggregation_order and len(custom_aggs):
                result = result.reindex(result_columns, axis=1)
            return result

        return aggregate_on_dict

    @classmethod
    def is_registered_implementation(cls, func: Callable) -> bool:
        """
        Check whether the passed `func` was registered as a TreeReduce implementation.

        Parameters
        ----------
        func : callable

        Returns
        -------
        bool
        """
        return callable(func) and hasattr(func, cls._GROUPBY_REDUCE_IMPL_FLAG)

    @classmethod
    def build_map_reduce_functions(
        cls,
        by,
        axis: int,
        groupby_kwargs: dict,
        map_func: Union[dict, Callable[..., pandas.DataFrame]],
        reduce_func: Union[dict, Callable[..., pandas.DataFrame]],
        agg_args: list,
        agg_kwargs: dict,
        drop: bool = False,
        method: Optional[str] = None,
        finalizer_fn: Callable[[pandas.DataFrame], pandas.DataFrame] = None,
    ) -> tuple[Callable, Callable]:
        """
        Bind appropriate arguments to map and reduce functions.

        Parameters
        ----------
        by : BaseQueryCompiler, column or index label, Grouper or list of such
            Object that determine groups.
        axis : {0, 1}
            Axis to group and apply aggregation function along. 0 means index axis
            when 1 means column axis.
        groupby_kwargs : dict
            Dictionary which carries arguments for pandas.DataFrame.groupby.
        map_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Map phase.
        reduce_func : dict or callable(pandas.DataFrameGroupBy) -> pandas.DataFrame
            Function to apply to the `GroupByObject` at the Reduce phase.
        agg_args : list-like
            Positional arguments to pass to the aggregation functions.
        agg_kwargs : dict
            Keyword arguments to pass to the aggregation functions.
        drop : bool, default: False
            Indicates whether or not by-data came from the `self` frame.
        method : str, optional
            Name of the GroupBy aggregation function. This is a hint to be able to do special casing.
        finalizer_fn : callable(pandas.DataFrame) -> pandas.DataFrame, optional
            A callable to execute at the end a groupby kernel against groupby result.

        Returns
        -------
        Tuple of callable
            Tuple of map and reduce functions with bound arguments.
        """
        # if by is a query compiler, then it will be broadcasted explicit via
        # groupby_reduce method of the modin frame and so we don't want secondary
        # implicit broadcastion via passing it as an function argument.
        if hasattr(by, "_modin_frame"):
            by = None

        def _map(
            df: pandas.DataFrame,
            other: Optional[pandas.DataFrame] = None,
            **kwargs: dict,
        ) -> pandas.DataFrame:
            def wrapper(
                df: pandas.DataFrame, other: Optional[pandas.DataFrame] = None
            ) -> pandas.DataFrame:
                return cls.map(
                    df,
                    other=other,
                    axis=axis,
                    by=by,
                    groupby_kwargs=groupby_kwargs.copy(),
                    map_func=map_func,
                    agg_args=agg_args,
                    agg_kwargs=agg_kwargs,
                    drop=drop,
                    **kwargs,
                )

            try:
                result = wrapper(df, other)
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except ValueError:
                result = wrapper(df.copy(), other if other is None else other.copy())
            return result

        def _reduce(df: pandas.DataFrame, **call_kwargs: dict) -> pandas.DataFrame:
            def wrapper(df: pandas.DataFrame):
                return cls.reduce(
                    df,
                    axis=axis,
                    groupby_kwargs=groupby_kwargs,
                    reduce_func=reduce_func,
                    agg_args=agg_args,
                    agg_kwargs=agg_kwargs,
                    drop=drop,
                    method=method,
                    finalizer_fn=finalizer_fn,
                    **call_kwargs,
                )

            try:
                result = wrapper(df)
            # This will happen with Arrow buffer read-only errors. We don't want to copy
            # all the time, so this will try to fast-path the code first.
            except ValueError:
                result = wrapper(df.copy())
            return result

        return _map, _reduce
