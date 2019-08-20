import ray
from itertools import groupby
import numpy as np
from operator import itemgetter
import pandas
from pandas.core.dtypes.cast import find_common_type
from pandas.core.index import ensure_index
from pandas.core.dtypes.common import (
    is_list_like,
    is_numeric_dtype,
    is_datetime_or_timedelta_dtype,
)

from .partition_manager import PandasOnRayFrameManager
from modin.backends.pandas.query_compiler import PandasQueryCompiler
from modin.error_message import ErrorMessage


class PandasOnRayData(object):

    _frame_mgr_cls = PandasOnRayFrameManager
    _query_compiler_cls = PandasQueryCompiler

    @property
    def __constructor__(self):
        return type(self)

    def __init__(self, partitions, index, columns, row_lengths=None, column_widths=None, dtypes=None):
        self._partitions = partitions
        self._index_cache = ensure_index(index)
        self._columns_cache = ensure_index(columns)
        self._row_lengths_cache = row_lengths
        self._column_widths_cache = column_widths
        self._dtypes = dtypes
        self._filter_empties()

    def _filter_empties(self):
        self._column_widths_cache = [w for w in self._column_widths if w > 0]
        self._row_lengths_cache = [r for r in self._row_lengths if r > 0]

        self._partitions = np.array(
            [[self._partitions[i][j]
              for j in range(len(self._partitions[i]))
              if j < len(self._column_widths)]
             for i in range(len(self._partitions))
             if i < len(self._row_lengths)])

    def _apply_index_objs(self, axis=None):
        self._filter_empties()
        if axis is None or axis == 0:
            cum_row_lengths = np.cumsum([0] + self._row_lengths)
        if axis is None or axis == 1:
            cum_col_widths = np.cumsum([0] + self._column_widths)

        if axis is None:
            def apply_idx_objs(df, idx, cols):
                df.index, df.columns = idx, cols
                return df

            self._partitions = np.array([[self._partitions[i][j].add_to_apply_calls(apply_idx_objs,
                                                              idx=self.index[slice(cum_row_lengths[i], cum_row_lengths[i + 1])],
                                                              cols=self.columns[slice(cum_col_widths[j], cum_col_widths[j + 1])])
                                 for j in range(len(self._partitions[i]))] for i in range(len(self._partitions))])
        elif axis == 0:
            def apply_idx_objs(df, idx):
                df.index = idx
                return df

            self._partitions = np.array([[self._partitions[i][j].add_to_apply_calls(apply_idx_objs,
                                                                           idx=self.index[
                                                                               slice(cum_row_lengths[i],
                                                                                     cum_row_lengths[i + 1])])
                                 for j in range(len(self._partitions[i]))] for i in
                                range(len(self._partitions))])
        elif axis == 1:
            def apply_idx_objs(df, cols):
                df.columns = cols
                return df

            self._partitions = np.array([[self._partitions[i][j].add_to_apply_calls(apply_idx_objs,
                                                                           cols=self.columns[slice(cum_col_widths[j],
                                                                                                   cum_col_widths[
                                                                                                       j + 1])])
                                 for j in range(len(self._partitions[i]))] for i in range(len(self._partitions))])
            ErrorMessage.catch_bugs_and_request_email(axis is not None and axis not in [0, 1])

    def mask(self, row_indices=None, row_numeric_idx=None, col_indices=None, col_numeric_idx=None):
        if row_indices is None and row_numeric_idx is None and col_indices is None and col_numeric_idx is None:
            return self.copy()
        if row_indices is not None:
            row_numeric_idx = self.index.get_indexer_for(row_indices)
        if row_numeric_idx is not None:
            row_partitions_list = self._get_dict_of_block_index(
                1, row_numeric_idx, ordered=True
            )
            new_row_lengths = [len(indices) for _, indices in row_partitions_list]
            new_index = self.index[row_numeric_idx]
        else:
            row_partitions_list = [
                (i, slice(None))
                for i in range(len(self._row_lengths))
            ]
            new_row_lengths = self._row_lengths
            new_index = self.index

        if col_indices is not None:
            col_numeric_idx = self.columns.get_indexer_for(col_indices)
        if col_numeric_idx is not None:
            col_partitions_list = self._get_dict_of_block_index(
                0, col_numeric_idx, ordered=True
            )
            new_col_widths = [len(indices) for _, indices in col_partitions_list]
            new_columns = self.columns[col_numeric_idx]
            new_dtypes = self.dtypes[col_numeric_idx]
        else:
            col_partitions_list = [
                (i, slice(None)) for i in range(len(self._column_widths))
            ]
            new_col_widths = self._column_widths
            new_columns = self.columns
            new_dtypes = self.dtypes
        new_partitions = np.array(
                [
                    [
                        self._partitions[row_idx][col_idx].mask(
                            row_internal_indices, col_internal_indices
                        )
                        for col_idx, col_internal_indices in col_partitions_list
                        if isinstance(col_internal_indices, slice) or len(col_internal_indices) > 0
                    ]
                    for row_idx, row_internal_indices in row_partitions_list
                    if isinstance(row_internal_indices, slice) or len(row_internal_indices) > 0
                ]
            )
        return self.__constructor__(new_partitions, new_index, new_columns, new_row_lengths, new_col_widths, new_dtypes)

    @property
    def _row_lengths(self):
        if self._row_lengths_cache is None:
            self._row_lengths_cache = [obj.length() for obj in self._partitions.T[0]]
        return self._row_lengths_cache

    @property
    def _column_widths(self):
        if self._column_widths_cache is None:
            self._column_widths_cache = [obj.width() for obj in self._partitions[0]]
        return self._column_widths_cache

    def copy(self):
        return self.__constructor__(
            self._partitions,
            self.index.copy(),
            self.columns.copy(),
            self._row_lengths,
            self._column_widths,
            self._dtypes,
        )

    @property
    def row_lengths(self):
        return self._row_lengths

    @property
    def column_widths(self):
        return self._column_widths

    @property
    def dtypes(self):
        if self._dtypes is None:
            self._dtypes = self._compute_dtypes()
        return self._dtypes

    def _compute_dtypes(self):
        def dtype_builder(df):
            return df.apply(lambda row: find_common_type(row.values), axis=0)

        map_func = self._build_mapreduce_func(0, lambda df: df.dtypes)
        reduce_func = self._build_mapreduce_func(0, dtype_builder)
        # For now we will use a pandas Series for the dtypes.
        if len(self.columns) > 0:
            dtypes = (
                self._full_reduce(0, map_func, reduce_func).to_pandas().iloc[0]
            )
        else:
            dtypes = pandas.Series([])
        # reset name to None because we use "__reduced__" internally
        dtypes.name = None
        return dtypes

    @classmethod
    def combine_dtypes(cls, dtypes_ids, column_names):
        # Compute dtypes by getting collecting and combining all of the partitions. The
        # reported dtypes from differing rows can be different based on the inference in
        # the limited data seen by each worker. We use pandas to compute the exact dtype
        # over the whole column for each column.
        dtypes = (
            pandas.concat(ray.get(dtypes_ids), axis=1)
                .apply(lambda row: find_common_type(row.values), axis=1)
                .squeeze(axis=0)
        )
        dtypes.index = column_names
        return dtypes

    def astype(self, col_dtypes):
        """Converts columns dtypes to given dtypes.

        Args:
            col_dtypes: Dictionary of {col: dtype,...} where col is the column
                name and dtype is a numpy dtype.

        Returns:
            DataFrame with updated dtypes.
        """
        columns = col_dtypes.keys()
        # Create Series for the updated dtypes
        new_dtypes = self.dtypes.copy()
        for i, column in enumerate(columns):
            dtype = col_dtypes[column]
            if (
                not isinstance(dtype, type(self.dtypes[column]))
                or dtype != self.dtypes[column]
            ):
                # Update the new dtype series to the proper pandas dtype
                try:
                    new_dtype = np.dtype(dtype)
                except TypeError:
                    new_dtype = dtype
                if dtype != np.int32 and new_dtype == np.int32:
                    new_dtype = np.dtype("int64")
                elif dtype != np.float32 and new_dtype == np.float32:
                    new_dtype = np.dtype("float64")
                new_dtypes[column] = new_dtype
        # Update partitions for each dtype that is updated

        def astype_builder(df):
            return df.astype({k: v for k, v in col_dtypes.items() if k in df})

        new_data = self._frame_mgr_cls.map_across_blocks(self._partitions, astype_builder)
        return self.__constructor__(new_data, self.index, self.columns, self._row_lengths, self._column_widths, new_dtypes)

    _index_cache = None
    _columns_cache = None

    def _validate_set_axis(self, new_labels, old_labels):
        new_labels = ensure_index(new_labels)
        old_len = len(old_labels)
        new_len = len(new_labels)
        if old_len != new_len:
            raise ValueError(
                "Length mismatch: Expected axis has %d elements, "
                "new values have %d elements" % (old_len, new_len)
            )
        return new_labels

    def _get_index(self):
        return self._index_cache

    def _get_columns(self):
        return self._columns_cache

    def _set_index(self, new_index):
        if self._index_cache is None:
            self._index_cache = ensure_index(new_index)
        else:
            new_index = self._validate_set_axis(new_index, self._index_cache)
            self._index_cache = new_index
        self._apply_index_objs(axis=0)

    def _set_columns(self, new_columns):
        if self._columns_cache is None:
            self._columns_cache = ensure_index(new_columns)
        else:
            new_columns = self._validate_set_axis(new_columns, self._columns_cache)
            self._columns_cache = new_columns
        self._apply_index_objs(axis=1)

    columns = property(_get_columns, _set_columns)
    index = property(_get_index, _set_index)

    # Metadata modification methods
    def add_prefix(self, prefix, axis):
        if axis == 1:
            new_columns = self.columns.map(lambda x: str(prefix) + str(x))
            if self._dtypes is not None:
                new_dtype_cache = self._dtypes.copy()
                new_dtype_cache.index = new_columns
            else:
                new_dtype_cache = None
            new_index = self.index
        else:
            new_index = self.index.map(lambda x: str(prefix) + str(x))
            new_columns = self.columns
            new_dtype_cache = self._dtypes
        new_data_obj = self.__constructor__(self._partitions, new_index, new_columns, self._row_lengths, self._column_widths, new_dtype_cache)
        new_data_obj._apply_index_objs(axis)
        return new_data_obj

    def add_suffix(self, suffix, axis):
        if axis == 1:
            new_columns = self.columns.map(lambda x: str(x) + str(suffix))
            if self._dtypes is not None:
                new_dtype_cache = self._dtypes.copy()
                new_dtype_cache.index = new_columns
            else:
                new_dtype_cache = None
            new_index = self.index
        else:
            new_index = self.index.map(lambda x: str(x) + str(suffix))
            new_columns = self.columns
            new_dtype_cache = self._dtypes
        new_data_obj = self.__constructor__(self._partitions, new_index, new_columns, self._row_lengths, self._column_widths, new_dtype_cache)
        new_data_obj._apply_index_objs(axis)
        return new_data_obj

    # END Metadata modification methods

    def _numeric_columns(self, include_bool=True):
        """Returns the numeric columns of the Manager.

        Returns:
            List of index names.
        """
        columns = []
        for col, dtype in zip(self.columns, self.dtypes):
            if is_numeric_dtype(dtype) and (
                include_bool or (not include_bool and dtype != np.bool_)
            ):
                columns.append(col)
        return columns

    def _get_dict_of_block_index(self, axis, indices, ordered=False):
        """Convert indices to a dict of block index to internal index mapping.

        Note: See `_get_blocks_containing_index` for primary usage. This method
            accepts a list of indices rather than just a single value, and uses
            `_get_blocks_containing_index`.

        Args:
            axis: The axis along which to get the indices
                (0 - columns, 1 - rows)
            indices: A list of global indices to convert.

        Returns
            For unordered: a dictionary of {block index: list of local indices}.
            For ordered: a list of tuples mapping block index: list of local indices.
        """
        if not ordered:
            indices = np.sort(indices)
        else:
            indices = np.array(indices)
        if not axis:
            # INT_MAX to make sure we don't try to compute on partitions that don't
            # exist.
            cumulative = np.array(
                np.append(self._column_widths[:-1], np.iinfo(np.int32).max)
            ).cumsum()
        else:
            cumulative = np.array(
                np.append(self._row_lengths[:-1], np.iinfo(np.int32).max)
            ).cumsum()

        def internal(block_idx, global_index):
            return (
                global_index
                if not block_idx
                else np.subtract(
                    global_index, cumulative[min(block_idx, len(cumulative) - 1) - 1]
                )
            )

        partition_ids = np.digitize(indices, cumulative)
        # If the output order doesn't matter or if the indices are monotonically
        # increasing, the computation is significantly simpler and faster than doing
        # the zip and groupby.
        if not ordered or np.all(np.diff(indices) > 0):
            count_for_each_partition = np.array(
                [(partition_ids == i).sum() for i in range(len(cumulative))]
            ).cumsum()
            # Compute the internal indices and pair those with the partition index.
            # If the first partition has any values we need to return, compute those
            # first to make the list comprehension easier. Otherwise, just append the
            # rest of the values to an empty list.
            if count_for_each_partition[0] > 0:
                first_partition_indices = [
                    (0, internal(0, indices[slice(count_for_each_partition[0])]))
                ]
            else:
                first_partition_indices = []
            partition_ids_with_indices = first_partition_indices + [
                (
                    i,
                    internal(
                        i,
                        indices[
                            slice(
                                count_for_each_partition[i - 1],
                                count_for_each_partition[i],
                            )
                        ],
                    ),
                )
                for i in range(1, len(count_for_each_partition))
                if count_for_each_partition[i] > count_for_each_partition[i - 1]
            ]
            return (
                dict(partition_ids_with_indices)
                if not ordered
                else partition_ids_with_indices
            )

        all_partitions_and_idx = zip(partition_ids, indices)
        # In ordered, we have to maintain the order of the list of indices provided.
        # This means that we need to return a list instead of a dictionary.
        return [
            (k, internal(k, [x for _, x in v]))
            for k, v in groupby(all_partitions_and_idx, itemgetter(0))
        ]

    def _join_index_objects(self, axis, other_index, how, sort):
        """Joins a pair of index objects (columns or rows) by a given strategy.

        Args:
            axis: The axis index object to join (0 for columns, 1 for index).
            other_index: The other_index to join on.
            how: The type of join to join to make (e.g. right, left).

        Returns:
            Joined indices.
        """
        if isinstance(other_index, list):
            joined_obj = self.columns if not axis else self.index
            # TODO: revisit for performance
            for obj in other_index:
                joined_obj = joined_obj.join(obj, how=how, sort=sort)
            return joined_obj
        if not axis:
            return self.columns.join(other_index, how=how, sort=sort)
        else:
            return self.index.join(other_index, how=how, sort=sort)

    # Internal methods
    # These methods are for building the correct answer in a modular way.
    # Please be careful when changing these!

    def _build_mapreduce_func(self, axis, func):
        def _map_reduce_func(df):
            series_result = func(df)
            if axis == 0 and isinstance(series_result, pandas.Series):
                # In the case of axis=0, we need to keep the shape of the data
                # consistent with what we have done. In the case of a reduction, the
                # data for axis=0 should be a single value for each column. By
                # transposing the data after we convert to a DataFrame, we ensure that
                # the columns of the result line up with the columns from the data.
                # axis=1 does not have this requirement because the index already will
                # line up with the index of the data based on how pandas creates a
                # DataFrame from a Series.
                return pandas.DataFrame(series_result).T
            return pandas.DataFrame(series_result)

        return _map_reduce_func

    def _full_axis_reduce(self, axis, func, alternate_index=None):
        """Applies map that reduce Manager to series but require knowledge of full axis.

        Args:
            func: Function to reduce the Manager by. This function takes in a Manager.
            axis: axis to apply the function to.
            alternate_index: If the resulting series should have an index
                different from the current query_compiler's index or columns.

        Return:
            Pandas series containing the reduced data.
        """
        func = self._build_mapreduce_func(axis, func)
        result = self._frame_mgr_cls.map_across_full_axis(axis, self._partitions, func)
        if axis == 0:
            columns = alternate_index if alternate_index is not None else self.columns
            return self.__constructor__(result, index=["__reduced__"], columns=columns, row_lengths=[1], column_widths=self.column_widths, dtypes=self.dtypes)
        else:
            index = alternate_index if alternate_index is not None else self.index
            new_dtypes = pandas.Series(np.full(1, find_common_type(self.dtypes.values)), index=["__reduced__"])
            return self.__constructor__(result, index=index, columns=["__reduced__"], row_lengths=self.row_lengths, column_widths=[1], dtypes=new_dtypes)

    def _full_reduce(self, axis, map_func, reduce_func=None):
        """Apply function that will reduce the data to a Pandas Series.

        Args:
            axis: 0 for columns and 1 for rows. Default is 0.
            map_func: Callable function to map the dataframe.
            reduce_func: Callable function to reduce the dataframe. If none,
                then apply map_func twice.

        Return:
            A new QueryCompiler object containing the results from map_func and
            reduce_func.
        """
        map_func = self._build_mapreduce_func(axis, map_func)
        if reduce_func is None:
            reduce_func = map_func
        else:
            reduce_func = self._build_mapreduce_func(axis, reduce_func)

        parts = self._frame_mgr_cls.map_across_blocks(self._partitions, map_func)
        final_parts = self._frame_mgr_cls.map_across_full_axis(axis, parts, reduce_func)
        if axis == 0:
            columns = self.columns
            index = ["__reduced__"]
            new_lengths = [1]
            new_widths = self._column_widths
        else:
            columns = ["__reduced__"]
            index = self.index
            new_lengths = self._row_lengths
            new_widths = [1]
        return self.__constructor__(
            final_parts, index, columns, new_lengths, new_widths,
        )

    def _map_partitions(self, func, dtypes=None):
        new_partitions = self._frame_mgr_cls.map_across_blocks(self._partitions, func)
        if dtypes == "copy":
            dtypes = self._dtypes
        elif dtypes is not None:
            dtypes = pandas.Series([np.dtype(dtypes)] * len(self.columns), index=self.columns)
        return self.__constructor__(
            new_partitions, self.index, self.columns, self._row_lengths, self._column_widths, dtypes=dtypes
        )

    def _map_across_full_axis(self, axis, func):
        new_partitions = self._frame_mgr_cls.map_across_full_axis(axis, self._partitions, func)
        return self.__constructor__(new_partitions, self.index, self.columns, self._row_lengths, self._column_widths)

    def _apply_full_axis(self, axis, func, new_index=None, new_columns=None, dtypes=None):
        new_partitions = self._frame_mgr_cls.map_across_full_axis(axis, self._partitions, func)
        # Index objects for new object creation. This is shorter than if..else
        if new_columns is None:
            new_columns = self._frame_mgr_cls.get_indices(1, new_partitions, lambda df: df.columns)
        if new_index is None:
            new_index = self._frame_mgr_cls.get_indices(1, new_partitions, lambda df: df.index)
        # Length objects for new object creation. This is shorter than if..else
        lengths_objs = {axis: None, axis ^ 1: [self._row_lengths, self._column_widths][axis ^ 1]}
        if dtypes == "copy":
            dtypes = self._dtypes
        elif dtypes is not None:
            dtypes = pandas.Series([np.dtype(dtypes)] * len(new_columns), index=new_columns)
        return self.__constructor__(new_partitions, new_index, new_columns, lengths_objs[0], lengths_objs[1], dtypes)

    def _apply_full_axis_select_indices(self, axis, func, apply_indices=None, numeric_indices=None, new_index=None, new_columns=None, keep_remaining=False):
        """Reduce Manger along select indices using function that needs full axis.

        Args:
            func: Callable that reduces the dimension of the object and requires full
                knowledge of the entire axis.
            axis: 0 for columns and 1 for rows. Defaults to 0.
            apply_indices: Index of the resulting QueryCompiler.

        Returns:
            A new QueryCompiler object with index or BaseFrameManager object.
        """
        assert apply_indices is not None or numeric_indices is not None
        # Convert indices to numeric indices
        old_index = self.index if axis else self.columns
        if apply_indices is not None:
            numeric_indices = old_index.get_indexer_for(apply_indices)
        dict_indices = self._get_dict_of_block_index(axis, numeric_indices)
        new_partitions = self._frame_mgr_cls.apply_func_to_select_indices_along_full_axis(
            axis, self._partitions, func, dict_indices, keep_remaining=keep_remaining
        )
        # TODO Infer columns and index from `keep_remaining` and `apply_indices`
        if new_index is None:
            new_index = self.index if axis == 1 else None
        if new_columns is None:
            new_columns = self.columns if axis == 0 else None
        # Length objects for new object creation. This is shorter than if..else
        lengths_objs = {
            axis: None if not keep_remaining else [self._row_lengths, self._column_widths][axis], axis ^ 1: [self._row_lengths, self._column_widths][axis ^ 1]
        }
        return self.__constructor__(new_partitions, new_index, new_columns, lengths_objs[0], lengths_objs[1])

    def _apply_select_indices(self, axis, func, apply_indices=None, row_indices=None, col_indices=None, new_index=None, new_columns=None, keep_remaining=False, item_to_distribute=None):
        # TODO Infer columns and index from `keep_remaining` and `apply_indices`
        if new_index is None:
            new_index = self.index if axis == 1 else None
        if new_columns is None:
            new_columns = self.columns if axis == 0 else None
        if axis is not None:
            assert apply_indices is not None
            # Convert indices to numeric indices
            old_index = self.index if axis else self.columns
            numeric_indices = old_index.get_indexer_for(apply_indices)
            dict_indices = self._get_dict_of_block_index(axis, numeric_indices)
            new_partitions = self._frame_mgr_cls.apply_func_to_select_indices(
                axis, self._partitions, func, dict_indices, keep_remaining=keep_remaining
            )
            # Length objects for new object creation. This is shorter than if..else
            lengths_objs = {axis: [len(apply_indices)] if not keep_remaining else [self._row_lengths, self._column_widths][axis], axis ^ 1: [self._row_lengths, self._column_widths][axis ^ 1]}
            return self.__constructor__(new_partitions, new_index, new_columns, lengths_objs[0], lengths_objs[1])
        else:
            assert row_indices is not None and col_indices is not None
            assert keep_remaining
            assert item_to_distribute is not None
            row_partitions_list = self._get_dict_of_block_index(1, row_indices).items()
            col_partitions_list = self._get_dict_of_block_index(0, col_indices).items()
            new_partitions = self._frame_mgr_cls.apply_func_to_indices_both_axis(self._partitions, func, row_partitions_list, col_partitions_list, item_to_distribute)
            return self.__constructor__(new_partitions, new_index, new_columns, self._row_lengths_cache, self._column_widths_cache)

    def _manual_repartition(self, axis, repartition_func, **kwargs):
        """This method applies all manual partitioning functions.

        Args:
            axis: The axis to shuffle data along.
            repartition_func: The function used to repartition data.

        Returns:
            A `BaseFrameManager` object.
        """
        func = self._prepare_method(repartition_func, **kwargs)
        return self.data.manual_shuffle(axis, func)

    def _copartition(self, axis, other, how, sort, force_repartition=False):
        """Copartition two QueryCompiler objects.

        Args:
            axis: The axis to copartition along.
            other: The other Query Compiler(s) to copartition against.
            how: How to manage joining the index object ("left", "right", etc.)
            sort: Whether or not to sort the joined index.
            force_repartition: Whether or not to force the repartitioning. By default,
                this method will skip repartitioning if it is possible. This is because
                reindexing is extremely inefficient. Because this method is used to
                `join` or `append`, it is vital that the internal indices match.

        Returns:
            A tuple (left query compiler, right query compiler list, joined index).
        """
        if isinstance(other, type(self)):
            other = [other]

        index_obj = (
            [o.index for o in other] if axis == 0 else [o.columns for o in other]
        )
        joined_index = self._join_index_objects(axis ^ 1, index_obj, how, sort)
        # We have to set these because otherwise when we perform the functions it may
        # end up serializing this entire object.
        left_old_idx = self.index if axis == 0 else self.columns
        right_old_idxes = index_obj

        # Start with this and we'll repartition the first time, and then not again.
        if not left_old_idx.equals(joined_index) or force_repartition:
            reindexed_self = self._frame_mgr_cls.map_across_full_axis(axis, self._partitions, lambda df: df.reindex(joined_index, axis=axis))
        else:
            reindexed_self = self._partitions
        reindexed_other_list = []

        for i in range(len(other)):
            if right_old_idxes[i].equals(joined_index) and not force_repartition:
                reindexed_other = other[i]._partitions
            else:
                reindexed_other = other[i]._frame_mgr_cls.map_across_full_axis(axis, other[i]._partitions, lambda df: df.reindex(joined_index, axis=axis))
            reindexed_other_list.append(reindexed_other)
        return reindexed_self, reindexed_other_list, joined_index

    def _binary_op(self, function, right_data, join_type="outer"):
        left_parts, right_parts, joined_index = self._copartition(
            0, right_data, join_type, sort=True
        )
        # unwrap list returned by `copartition`.
        right_parts = right_parts[0]
        new_data = self._frame_mgr_cls.inter_data_operation(
            1, left_parts, lambda l, r: function(l, r), right_parts
        )
        new_columns = self.columns.join(right_data.columns, how=join_type)
        return self.__constructor__(new_data, self.index, new_columns, None, None)

    def _concat(self, axis, others, how, sort):
        #TODO Update to no longer force repartition
        # Requires pruning of the partitions after they have been changed
        left_parts, right_parts, joined_index = self._copartition(
            axis ^ 1,
            others,
            how,
            sort,
            force_repartition=True,
        )
        new_partitions = self._frame_mgr_cls.concat(axis, left_parts, right_parts)
        if axis == 0:
            new_index = self.index.append([other.index for other in others])
            new_columns = joined_index
        else:
            new_columns = self.columns.append([other.columns for other in others])
            new_index = joined_index
        return self.__constructor__(new_partitions, new_index, new_columns)

    @classmethod
    def from_pandas(cls, df):
        """Improve simple Pandas DataFrame to an advanced and superior Modin DataFrame.

        Args:
            cls: DataManger object to convert the DataFrame to.
            df: Pandas DataFrame object.
            block_partitions_cls: BlockParitions object to store partitions

        Returns:
            Returns QueryCompiler containing data from the Pandas DataFrame.
        """
        new_index = df.index
        new_columns = df.columns
        new_dtypes = df.dtypes
        new_data, new_lengths, new_widths = cls._frame_mgr_cls.from_pandas(df, True)
        return cls(new_data, new_index, new_columns, new_lengths, new_widths, dtypes=new_dtypes)

    def to_pandas(self):
        """Converts Modin DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame of the QueryCompiler.
        """
        df = self._frame_mgr_cls.to_pandas(self._partitions)
        if df.empty:
            if len(self.columns) != 0:
                df = pandas.DataFrame(columns=self.columns).astype(self.dtypes)
            else:
                df = pandas.DataFrame(columns=self.columns, index=self.index)
        df.index.name = self.index.name
        return df

    def to_numpy(self):
        return self._frame_mgr_cls.to_numpy(self._partitions)

    def transpose(self):
        new_partitions = np.array([[part.add_to_apply_calls(pandas.DataFrame.transpose) for part in row] for row in self._partitions]).T
        new_dtypes = pandas.Series(np.full(len(self.index), find_common_type(self.dtypes.values)), index=self.index)
        return self.__constructor__(new_partitions, self.columns, self.index, self._column_widths, self._row_lengths, dtypes=new_dtypes)

    # Head/Tail/Front/Back
    @staticmethod
    def _compute_lengths(lengths_list, n, from_back=False):
        if not from_back:
            idx = np.digitize(n, np.cumsum(lengths_list))
            if idx == 0:
                return [n]
            return [
                lengths_list[i] if i < idx else n - sum(lengths_list[:i])
                for i in range(len(lengths_list)) if i <= idx
            ]
        else:
            lengths_list = [i for i in lengths_list if i > 0]
            idx = np.digitize(sum(lengths_list) - n, np.cumsum(lengths_list))
            if idx == len(lengths_list) - 1:
                return [n]
            return [
                   lengths_list[i] if i > idx else n - sum(lengths_list[i + 1:])
                   for i in range(len(lengths_list)) if i >= idx
               ]

    def head(self, n):
        """Returns the first n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the first n rows of the original QueryCompiler.
        """
        # We grab the front if it is transposed and flag as transposed so that
        # we are not physically updating the data from this manager. This
        # allows the implementation to stay modular and reduces data copying.
        if n < 0:
            n = max(0, len(self.index) + n)
        new_row_lengths = self._compute_lengths(self._row_lengths, n)
        new_partitions = self._frame_mgr_cls.take(0, self._partitions, self._row_lengths, n)
        return self.__constructor__(
            new_partitions, self.index[:n], self.columns, new_row_lengths, self._column_widths, self.dtypes,
        )

    def tail(self, n):
        """Returns the last n rows.

        Args:
            n: Integer containing the number of rows to return.

        Returns:
            QueryCompiler containing the last n rows of the original QueryCompiler.
        """
        # See head for an explanation of the transposed behavior
        if n < 0:
            n = max(0, len(self.index) + n)
        new_row_lengths = self._compute_lengths(self._row_lengths, n, from_back=True)
        new_partitions = self._frame_mgr_cls.take(0, self._partitions, self._row_lengths, -n)
        return self.__constructor__(
            new_partitions, self.index[-n:], self.columns, new_row_lengths, self._column_widths, self.dtypes,
        )

    def front(self, n):
        """Returns the first n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the first n columns of the original QueryCompiler.
        """
        new_col_lengths = self._compute_lengths(self._column_widths, n)
        new_partitions = self._frame_mgr_cls.take(1, self._partitions, self._column_widths, n)
        return self.__constructor__(
            new_partitions, self.index, self.columns[:n], self._row_lengths, new_col_lengths, self.dtypes[:n],
        )

    def back(self, n):
        """Returns the last n columns.

        Args:
            n: Integer containing the number of columns to return.

        Returns:
            QueryCompiler containing the last n columns of the original QueryCompiler.
        """
        new_col_lengths = self._compute_lengths(self._column_widths, n, from_back=True)
        new_partitions = self._frame_mgr_cls.take(1, self._partitions, self._column_widths, -n)
        return self.__constructor__(
            new_partitions, self.index, self.columns[-n:], self._row_lengths, new_col_lengths, self.dtypes[n:],
        )

    # End Head/Tail/Front/Back
