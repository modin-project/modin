import ray
import numpy as np
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

    def __init__(self, partitions, index, columns, row_lengths, column_widths, dtypes=None):
        self._partitions = partitions
        self._index_cache = index
        self._columns_cache = columns
        self._row_lengths = row_lengths
        self._column_widths = column_widths
        self._dtypes = dtypes
        self._filter_empties()

    def _filter_empties(self):
        self._partitions = np.array(
            [[self._partitions[i][j]
              for j in range(len(self._partitions[i]))
              if j < len(self._column_widths) and self._column_widths[j] > 0]
             for i in range(len(self._partitions))
             if i < len(self._row_lengths) and self._row_lengths[i] > 0])

        self._column_widths = [w for w in self._column_widths if w > 0]
        self._row_lengths = [r for r in self._row_lengths if r > 0]

    def _apply_index_objs(self, axis=None):
        self._filter_empties()
        if axis is None or axis == 0:
            cum_row_lengths = np.cumsum([0] + self._row_lengths)
        if axis is None or axis == 1:
            cum_col_widths = np.cumsum([0] + self._column_widths)

        if axis is None:
            def apply_idx_objs(df, idx, col):
                df.index, df.columns = idx, col
                return df

            for i in range(len(self._partitions)):
                for j in range(len(self._partitions[i])):
                    self._partitions[i][j].call_queue += [(apply_idx_objs, {
                        "idx": self.index[slice(cum_row_lengths[i], cum_row_lengths[i + 1])],
                        "col": self.columns[slice(cum_col_widths[j], cum_col_widths[j + 1])]})]
        elif axis == 0:
            def apply_idx_objs(df, idx):
                df.index = idx
                return df

            for i in range(len(self._partitions)):
                for j in range(len(self._partitions[i])):
                    self._partitions[i][j].call_queue += [(apply_idx_objs, {
                        "idx": self.index[slice(cum_row_lengths[i], cum_row_lengths[i + 1])]})]

        elif axis == 1:
            def apply_idx_objs(df, cols):
                df.cols = cols
                return df

            for i in range(len(self._partitions)):
                for j in range(len(self._partitions[i])):
                    self._partitions[i][j].call_queue += [(apply_idx_objs, {
                        "col": self.columns[slice(cum_col_widths[j], cum_col_widths[j + 1])]})]
        else:
            ErrorMessage.catch_bugs_and_request_email(axis is not None and axis not in [0, 1])

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

    def _join_index_objects(self, axis, other_index, how, sort=True):
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
                joined_obj = joined_obj.join(obj, how=how)

            return joined_obj
        if not axis:
            return self.columns.join(other_index, how=how, sort=sort)
        else:
            return self.index.join(other_index, how=how, sort=sort)

    # Internal methods
    # These methods are for building the correct answer in a modular way.
    # Please be careful when changing these!

    def _build_mapreduce_func(self, axis, func, **kwargs):
        def _map_reduce_func(df):
            series_result = func(df, **kwargs)
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
            return self.__constructor__(result.partitions, index=["__reduced__"], columns=columns, row_lengths=[1], column_widths=self.column_widths, dtypes=self.dtypes)
        else:
            index = alternate_index if alternate_index is not None else self.index
            new_dtypes = pandas.Series(np.full(1, find_common_type(self.dtypes.values)), index=["__reduced__"])
            return self.__constructor__(result.partitions, index=index, columns=["__reduced__"], row_lengths=self.row_lengths, column_widths=[1], dtypes=new_dtypes)

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
            dtypes = self.dtypes
        elif dtypes is not None:
            dtypes = pandas.Series([dtypes] * len(self.columns), index=self.columns)
        return self.__constructor__(
            new_partitions, self.index, self.columns, self._row_lengths, self._column_widths, dtypes=dtypes
        )

    def _map_across_full_axis(self, axis, func):
        new_partitions = self._frame_mgr_cls.map_across_full_axis(axis, self._partitions, func)
        first_idx, first_col = ray.get(new_partitions[0][0].apply(lambda df: (df.index, df.columns)).oid)
        idx_parts = ray.get([part.apply(lambda df: df.index).oid for part in new_partitions.T[0][1:]])
        col_parts = ray.get([part.apply(lambda df: df.columns).oid for part in new_partitions[0][1:]])
        new_lengths = [len(obj) for obj in [first_idx] + idx_parts]
        new_widths = [len(obj) for obj in [first_col] + col_parts]
        new_idx = first_idx.append(idx_parts)
        new_cols = first_col.append(col_parts)
        return self.__constructor__(new_partitions, new_idx, new_cols, new_lengths, new_widths)

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
        new_data = cls._frame_mgr_cls.from_pandas(df)
        return cls(new_data, new_index, new_columns, dtypes=new_dtypes)

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
        return df

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
