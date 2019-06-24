from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas
from pandas.compat import string_types
from pandas.core.common import apply_if_callable, is_bool_indexer
from pandas.core.dtypes.common import (
    infer_dtype_from_object,
    is_dict_like,
    is_list_like,
    is_numeric_dtype,
)
from pandas.core.index import ensure_index_from_sequences
from pandas.core.indexing import check_bool_indexer
from pandas.util._validators import validate_bool_kwarg

import itertools
import functools
import numpy as np
import sys
import warnings

from modin.error_message import ErrorMessage
from .utils import from_pandas, to_pandas, _inherit_docstrings
from .iterator import PartitionIterator
from .series import Series
from .base import BasePandasDataset


@_inherit_docstrings(
    pandas.DataFrame, excluded=[pandas.DataFrame, pandas.DataFrame.__init__]
)
class DataFrame(BasePandasDataset):
    def __init__(
        self,
        data=None,
        index=None,
        columns=None,
        dtype=None,
        copy=False,
        query_compiler=None,
    ):
        """Distributed DataFrame object backed by Pandas dataframes.

        Args:
            data (numpy ndarray (structured or homogeneous) or dict):
                Dict can contain Series, arrays, constants, or list-like
                objects.
            index (pandas.Index, list, ObjectID): The row index for this
                DataFrame.
            columns (pandas.Index): The column names for this DataFrame, in
                pandas Index object.
            dtype: Data type to force. Only a single dtype is allowed.
                If None, infer
            copy (boolean): Copy data from inputs.
                Only affects DataFrame / 2d ndarray input.
            query_compiler: A query compiler object to manage distributed computation.
        """
        if isinstance(data, (DataFrame, Series)):
            self._query_compiler = data._query_compiler
            if isinstance(data, Series) and data.name is None:
                self.columns = [0]
            else:
                data._add_sibling(self)
        # Check type of data and use appropriate constructor
        elif query_compiler is None:
            warnings.warn(
                "Distributing {} object. This may take some time.".format(type(data))
            )
            if is_list_like(data) and not is_dict_like(data):
                data = [
                    obj._to_pandas() if isinstance(obj, Series) else obj for obj in data
                ]
            elif is_dict_like(data) and not isinstance(
                data, (pandas.Series, Series, pandas.DataFrame, DataFrame)
            ):
                data = {
                    k: v._to_pandas() if isinstance(v, Series) else v
                    for k, v in data.items()
                }
            pandas_df = pandas.DataFrame(
                data=data, index=index, columns=columns, dtype=dtype, copy=copy
            )
            self._query_compiler = from_pandas(pandas_df)._query_compiler
        else:
            self._query_compiler = query_compiler

    def __repr__(self):
        num_rows = pandas.get_option("max_rows") or 60
        num_cols = pandas.get_option("max_columns") or 20

        result = repr(self._build_repr_df(num_rows, num_cols))
        if len(self.index) > num_rows or len(self.columns) > num_cols:
            # The split here is so that we don't repr pandas row lengths.
            return result.rsplit("\n\n", 1)[0] + "\n\n[{0} rows x {1} columns]".format(
                len(self.index), len(self.columns)
            )
        else:
            return result

    def _repr_html_(self):  # pragma: no cover
        """repr function for rendering in Jupyter Notebooks like Pandas
        Dataframes.

        Returns:
            The HTML representation of a Dataframe.
        """
        num_rows = pandas.get_option("max_rows") or 60
        num_cols = pandas.get_option("max_columns") or 20

        # We use pandas _repr_html_ to get a string of the HTML representation
        # of the dataframe.
        result = self._build_repr_df(num_rows, num_cols)._repr_html_()
        if len(self.index) > num_rows or len(self.columns) > num_cols:
            # We split so that we insert our correct dataframe dimensions.
            return result.split("<p>")[
                0
            ] + "<p>{0} rows x {1} columns</p>\n</div>".format(
                len(self.index), len(self.columns)
            )
        else:
            return result

    def _get_columns(self):
        """Get the columns for this DataFrame.

        Returns:
            The union of all indexes across the partitions.
        """
        return self._query_compiler.columns

    def _set_columns(self, new_columns):
        """Set the columns for this DataFrame.

        Args:
            new_columns: The new index to set this
        """
        self._query_compiler.columns = new_columns

    columns = property(_get_columns, _set_columns)

    def _validate_eval_query(self, expr, **kwargs):
        """Helper function to check the arguments to eval() and query()

        Args:
            expr: The expression to evaluate. This string cannot contain any
                Python statements, only Python expressions.
        """
        if isinstance(expr, str) and expr == "":
            raise ValueError("expr cannot be an empty string")

        if isinstance(expr, str) and "@" in expr:
            ErrorMessage.not_implemented("Local variables not yet supported in eval.")

        if isinstance(expr, str) and "not" in expr:
            if "parser" in kwargs and kwargs["parser"] == "python":
                ErrorMessage.not_implemented(
                    "'Not' nodes are not implemented."
                )  # pragma: no cover

    @property
    def ndim(self):
        """Get the number of dimensions for this DataFrame.

        Returns:
            The number of dimensions for this DataFrame.
        """
        # DataFrames have an invariant that requires they be 2 dimensions.
        return 2

    @property
    def ftypes(self):
        """Get the ftypes for this DataFrame.

        Returns:
            The ftypes for this DataFrame.
        """
        # The ftypes are common across all partitions.
        # The first partition will be enough.
        dtypes = self.dtypes.copy()
        ftypes = ["{0}:dense".format(str(dtype)) for dtype in dtypes.values]
        result = pandas.Series(ftypes, index=self.columns)
        return result

    def drop_duplicates(self, subset=None, keep="first", inplace=False):
        return super(DataFrame, self).drop_duplicates(
            subset=subset, keep=keep, inplace=inplace
        )

    @property
    def dtypes(self):
        """Get the dtypes for this DataFrame.

        Returns:
            The dtypes for this DataFrame.
        """
        return self._query_compiler.dtypes

    def duplicated(self, subset=None, keep="first"):
        return super(DataFrame, self).duplicated(subset=subset, keep=keep)

    @property
    def empty(self):
        """Determines if the DataFrame is empty.

        Returns:
            True if the DataFrame is empty.
            False otherwise.
        """
        return len(self.columns) == 0 or len(self.index) == 0

    @property
    def axes(self):
        """Get the axes for the DataFrame.

        Returns:
            The axes for the DataFrame.
        """
        return [self.index, self.columns]

    @property
    def shape(self):
        """Get the size of each of the dimensions in the DataFrame.

        Returns:
            A tuple with the size of each dimension as they appear in axes().
        """
        return len(self.index), len(self.columns)

    def add_prefix(self, prefix):
        """Add a prefix to each of the column names.

        Returns:
            A new DataFrame containing the new column names.
        """
        return DataFrame(query_compiler=self._query_compiler.add_prefix(prefix))

    def add_suffix(self, suffix):
        """Add a suffix to each of the column names.

        Returns:
            A new DataFrame containing the new column names.
        """
        return DataFrame(query_compiler=self._query_compiler.add_suffix(suffix))

    def applymap(self, func):
        """Apply a function to a DataFrame elementwise.

        Args:
            func (callable): The function to apply.
        """
        if not callable(func):
            raise ValueError("'{0}' object is not callable".format(type(func)))
        ErrorMessage.non_verified_udf()
        return DataFrame(query_compiler=self._query_compiler.applymap(func))

    def apply(
        self,
        func,
        axis=0,
        broadcast=None,
        raw=False,
        reduce=None,
        result_type=None,
        convert_dtype=True,
        args=(),
        **kwds
    ):
        axis = self._get_axis_number(axis)
        query_compiler = super(DataFrame, self).apply(
            func,
            axis=axis,
            broadcast=broadcast,
            raw=raw,
            reduce=reduce,
            result_type=result_type,
            convert_dtype=convert_dtype,
            args=args,
            **kwds
        )
        if not isinstance(query_compiler, type(self._query_compiler)):
            return query_compiler
        # This is the simplest way to determine the return type, but there are checks
        # in pandas that verify that some results are created. This is a challenge for
        # empty DataFrames, but fortunately they only happen when the `func` type is
        # a list or a dictionary, which means that the return type won't change from
        # type(self), so we catch that error and use `self.__name__` for the return
        # type.
        try:
            if axis == 0:
                init_kwargs = {"index": self.index}
            else:
                init_kwargs = {"columns": self.columns}
            return_type = type(
                getattr(pandas, self.__name__)(**init_kwargs).apply(
                    func,
                    axis=axis,
                    broadcast=broadcast,
                    raw=raw,
                    reduce=reduce,
                    result_type=result_type,
                )
            ).__name__
        except Exception:
            return_type = self.__name__
        if return_type not in ["DataFrame", "Series"]:
            return query_compiler.to_pandas().squeeze()
        else:
            result = getattr(sys.modules[self.__module__], return_type)(
                query_compiler=query_compiler
            )
            if hasattr(result, "name"):
                if axis == 0 and result.name == self.index[0]:
                    result.name = None
                elif axis == 1 and result.name == self.columns[0]:
                    result.name = None
            return result

    def get_value(self, index, col, takeable=False):
        return self._default_to_pandas(
            pandas.DataFrame.get_value, index, col, takeable=takeable
        )

    def groupby(
        self,
        by=None,
        axis=0,
        level=None,
        as_index=True,
        sort=True,
        group_keys=True,
        squeeze=False,
        observed=False,
        **kwargs
    ):
        """Apply a groupby to this DataFrame. See _groupby() remote task.
        Args:
            by: The value to groupby.
            axis: The axis to groupby.
            level: The level of the groupby.
            as_index: Whether or not to store result as index.
            sort: Whether or not to sort the result by the index.
            group_keys: Whether or not to group the keys.
            squeeze: Whether or not to squeeze.
        Returns:
            A new DataFrame resulting from the groupby.
        """
        axis = self._get_axis_number(axis)
        idx_name = None
        if callable(by):
            by = by(self.index)
        elif isinstance(by, string_types):
            idx_name = by
            by = self.__getitem__(by)._query_compiler
        elif is_list_like(by):
            if isinstance(by, Series):
                idx_name = by.name
                by = by.values
            mismatch = (
                len(by) != len(self) if axis == 0 else len(by) != len(self.columns)
            )
            if mismatch and all(obj in self for obj in by):
                # In the future, we will need to add logic to handle this, but for now
                # we default to pandas in this case.
                pass
            elif mismatch:
                raise KeyError(next(x for x in by if x not in self))

        from .groupby import DataFrameGroupBy

        return DataFrameGroupBy(
            self,
            by,
            axis,
            level,
            as_index,
            sort,
            group_keys,
            squeeze,
            idx_name,
            observed=observed,
            **kwargs
        )

    def _reduce_dimension(self, query_compiler):
        return Series(query_compiler=query_compiler)

    def keys(self):
        """Get the info axis for the DataFrame.

        Returns:
            A pandas Index for this DataFrame.
        """
        return self.columns

    def transpose(self, *args, **kwargs):
        """Transpose columns and rows for the DataFrame.

        Returns:
            A new DataFrame transposed from this DataFrame.
        """
        return DataFrame(query_compiler=self._query_compiler.transpose(*args, **kwargs))

    T = property(transpose)

    def add(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).add(
            other, axis=axis, level=level, fill_value=fill_value
        )

    def append(self, other, ignore_index=False, verify_integrity=False, sort=None):
        """Append another DataFrame/list/Series to this one.

        Args:
            other: The object to append to this.
            ignore_index: Ignore the index on appending.
            verify_integrity: Verify the integrity of the index on completion.

        Returns:
            A new DataFrame containing the concatenated values.
        """
        if isinstance(other, (Series, dict)):
            if isinstance(other, dict):
                other = Series(other)
            if other.name is None and not ignore_index:
                raise TypeError(
                    "Can only append a Series if ignore_index=True"
                    " or if the Series has a name"
                )
            if other.name is not None:
                # other must have the same index name as self, otherwise
                # index name will be reset
                name = other.name
                # We must transpose here because a Series becomes a new row, and the
                # structure of the query compiler is currently columnar
                other = other._query_compiler.transpose()
                other.index = pandas.Index([name], name=self.index.name)
            else:
                # See note above about transpose
                other = other._query_compiler.transpose()
        elif isinstance(other, list):
            if not all(isinstance(o, BasePandasDataset) for o in other):
                other = DataFrame(pandas.DataFrame(other))._query_compiler
            else:
                other = [obj._query_compiler for obj in other]
        else:
            other = other._query_compiler

        # If ignore_index is False, by definition the Index will be correct.
        # We also do this first to ensure that we don't waste compute/memory.
        if verify_integrity and not ignore_index:
            appended_index = (
                self.index.append(other.index)
                if not isinstance(other, list)
                else self.index.append([o.index for o in other])
            )
            is_valid = next((False for idx in appended_index.duplicated() if idx), True)
            if not is_valid:
                raise ValueError(
                    "Indexes have overlapping values: {}".format(
                        appended_index[appended_index.duplicated()]
                    )
                )

        query_compiler = self._query_compiler.concat(
            0, other, ignore_index=ignore_index, sort=sort
        )
        return DataFrame(query_compiler=query_compiler)

    def assign(self, **kwargs):
        return self._default_to_pandas(pandas.DataFrame.assign, **kwargs)

    def boxplot(
        self,
        column=None,
        by=None,
        ax=None,
        fontsize=None,
        rot=0,
        grid=True,
        figsize=None,
        layout=None,
        return_type=None,
        **kwds
    ):
        return to_pandas(self).boxplot(
            column=column,
            by=by,
            ax=ax,
            fontsize=fontsize,
            rot=rot,
            grid=grid,
            figsize=figsize,
            layout=layout,
            return_type=return_type,
            **kwds
        )

    def combine(self, other, func, fill_value=None, overwrite=True):
        return super(DataFrame, self).combine(
            other, func, fill_value=fill_value, overwrite=overwrite
        )

    def convert_objects(
        self,
        convert_dates=True,
        convert_numeric=False,
        convert_timedeltas=True,
        copy=True,
    ):
        return self._default_to_pandas(
            pandas.DataFrame.convert_objects,
            convert_dates=convert_dates,
            convert_numeric=convert_numeric,
            convert_timedeltas=convert_timedeltas,
            copy=copy,
        )

    def corr(self, method="pearson", min_periods=1):
        return self._default_to_pandas(
            pandas.DataFrame.corr, method=method, min_periods=min_periods
        )

    def corrwith(self, other, axis=0, drop=False, method="pearson"):
        if isinstance(other, DataFrame):
            other = other._query_compiler.to_pandas()
        return self._default_to_pandas(
            pandas.DataFrame.corrwith, other, axis=axis, drop=drop, method=method
        )

    def cov(self, min_periods=None):
        return self._default_to_pandas(pandas.DataFrame.cov, min_periods=min_periods)

    def eq(self, other, axis="columns", level=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).eq(other, axis=axis, level=level)

    def equals(self, other):
        """
        Checks if other DataFrame is elementwise equal to the current one

        Returns:
            Boolean: True if equal, otherwise False
        """
        if isinstance(other, pandas.DataFrame):
            # Copy into a Modin DataFrame to simplify logic below
            other = DataFrame(other)
        return (
            self.index.equals(other.index)
            and self.columns.equals(other.columns)
            and self.eq(other).all().all()
        )

    def eval(self, expr, inplace=False, **kwargs):
        """Evaluate a Python expression as a string using various backends.
        Args:
            expr: The expression to evaluate. This string cannot contain any
                Python statements, only Python expressions.

            parser: The parser to use to construct the syntax tree from the
                expression. The default of 'pandas' parses code slightly
                different than standard Python. Alternatively, you can parse
                an expression using the 'python' parser to retain strict
                Python semantics. See the enhancing performance documentation
                for more details.

            engine: The engine used to evaluate the expression.

            truediv: Whether to use true division, like in Python >= 3

            local_dict: A dictionary of local variables, taken from locals()
                by default.

            global_dict: A dictionary of global variables, taken from
                globals() by default.

            resolvers: A list of objects implementing the __getitem__ special
                method that you can use to inject an additional collection
                of namespaces to use for variable lookup. For example, this is
                used in the query() method to inject the index and columns
                variables that refer to their respective DataFrame instance
                attributes.

            level: The number of prior stack frames to traverse and add to
                the current scope. Most users will not need to change this
                parameter.

            target: This is the target object for assignment. It is used when
                there is variable assignment in the expression. If so, then
                target must support item assignment with string keys, and if a
                copy is being returned, it must also support .copy().

            inplace: If target is provided, and the expression mutates target,
                whether to modify target inplace. Otherwise, return a copy of
                target with the mutation.
        Returns:
            ndarray, numeric scalar, DataFrame, Series
        """
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")
        new_query_compiler = self._query_compiler.eval(expr, **kwargs)
        return_type = type(
            pandas.DataFrame(columns=self.columns)
            .astype(self.dtypes)
            .eval(expr, **kwargs)
        ).__name__
        if return_type == self.__name__:
            return self._create_or_update_from_compiler(new_query_compiler, inplace)
        else:
            if inplace:
                raise ValueError("Cannot operate inplace if there is no assignment")
            return getattr(sys.modules[self.__module__], return_type)(
                query_compiler=new_query_compiler
            )

    def floordiv(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).floordiv(
            other, axis=axis, level=level, fill_value=None
        )

    @classmethod
    def from_dict(
        cls, data, orient="columns", dtype=None, columns=None
    ):  # pragma: no cover
        ErrorMessage.default_to_pandas("`from_dict`")
        return from_pandas(
            pandas.DataFrame.from_dict(
                data, orient=orient, dtype=dtype, columns=columns
            )
        )

    @classmethod
    def from_items(cls, items, columns=None, orient="columns"):  # pragma: no cover
        ErrorMessage.default_to_pandas("`from_items`")
        return from_pandas(
            pandas.DataFrame.from_items(items, columns=columns, orient=orient)
        )

    @classmethod
    def from_records(
        cls,
        data,
        index=None,
        exclude=None,
        columns=None,
        coerce_float=False,
        nrows=None,
    ):  # pragma: no cover
        ErrorMessage.default_to_pandas("`from_records`")
        return from_pandas(
            pandas.DataFrame.from_records(
                data,
                index=index,
                exclude=exclude,
                columns=columns,
                coerce_float=coerce_float,
                nrows=nrows,
            )
        )

    def ge(self, other, axis="columns", level=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).ge(other, axis=axis, level=level)

    def gt(self, other, axis="columns", level=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).gt(other, axis=axis, level=level)

    def head(self, n=5):
        if n == 0:
            return DataFrame(columns=self.columns)
        return super(DataFrame, self).head(n)

    def hist(
        self,
        column=None,
        by=None,
        grid=True,
        xlabelsize=None,
        xrot=None,
        ylabelsize=None,
        yrot=None,
        ax=None,
        sharex=False,
        sharey=False,
        figsize=None,
        layout=None,
        bins=10,
        **kwds
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.hist,
            column=column,
            by=by,
            grid=grid,
            xlabelsize=xlabelsize,
            xrot=xrot,
            ylabelsize=ylabelsize,
            yrot=yrot,
            ax=ax,
            sharex=sharex,
            sharey=sharey,
            figsize=figsize,
            layout=layout,
            bins=bins,
            **kwds
        )

    def info(
        self, verbose=None, buf=None, max_cols=None, memory_usage=None, null_counts=None
    ):
        """Print a concise summary of a DataFrame, which includes the index
        dtype and column dtypes, non-null values and memory usage.

        Args:
            verbose (bool, optional): Whether to print the full summary. Defaults
                to true

            buf (writable buffer): Where to send output. Defaults to sys.stdout

            max_cols (int, optional): When to switch from verbose to truncated
                output. By defualt, this is 100.

            memory_usage (bool, str, optional): Specifies whether the total memory
                usage of the DataFrame elements (including index) should be displayed.
                True always show memory usage. False never shows memory usage. A value
                of 'deep' is equivalent to "True with deep introspection". Memory usage
                is shown in human-readable units (base-2 representation). Without deep
                introspection a memory estimation is made based in column dtype and
                number of rows assuming values consume the same memory amount for
                corresponding dtypes. With deep memory introspection, a real memory
                usage calculation is performed at the cost of computational resources.
                Defaults to True.

            null_counts (bool, optional): Whetehr to show the non-null counts. By
                default, this is shown only when the frame is smaller than 100 columns
                and 1690785 rows. A value of True always shows the counts and False
                never shows the counts.

        Returns:
            Prints the summary of a DataFrame and returns None.
        """
        # We will default to pandas because it will be faster than doing two passes
        # over the data
        buf = sys.stdout if not buf else buf
        import io

        with io.StringIO() as tmp_buf:
            self._default_to_pandas(
                pandas.DataFrame.info,
                verbose=verbose,
                buf=tmp_buf,
                max_cols=max_cols,
                memory_usage=memory_usage,
                null_counts=null_counts,
            )
            result = tmp_buf.getvalue()
            result = result.replace(
                "pandas.core.frame.DataFrame", "modin.pandas.dataframe.DataFrame"
            )
            buf.write(result)
        return None

    def insert(self, loc, column, value, allow_duplicates=False):
        """Insert column into DataFrame at specified location.
        Args:
            loc (int): Insertion index. Must verify 0 <= loc <= len(columns).
            column (hashable object): Label of the inserted column.
            value (int, Series, or array-like): The values to insert.
            allow_duplicates (bool): Whether to allow duplicate column names.
        """
        if isinstance(value, (DataFrame, pandas.DataFrame)):
            if len(value.columns) != 1:
                raise ValueError("Wrong number of items passed 2, placement implies 1")
            value = value.iloc[:, 0]
        if len(self.index) == 0:
            if isinstance(value, Series):
                # TODO: Remove broadcast of Series
                value = value._to_pandas()
            try:
                value = pandas.Series(value)
            except (TypeError, ValueError, IndexError):
                raise ValueError(
                    "Cannot insert into a DataFrame with no defined index "
                    "and a value that cannot be converted to a "
                    "Series"
                )
            new_index = value.index.copy()
            new_columns = self.columns.insert(loc, column)
            new_query_compiler = DataFrame(
                value, index=new_index, columns=new_columns
            )._query_compiler
        elif len(self.columns) == 0 and loc == 0:
            new_query_compiler = DataFrame(
                data=value, columns=[column], index=self.index
            )._query_compiler
        else:
            if not is_list_like(value):
                value = np.full(len(self.index), value)
            if not isinstance(value, pandas.Series) and len(value) != len(self.index):
                raise ValueError("Length of values does not match length of index")
            if not allow_duplicates and column in self.columns:
                raise ValueError("cannot insert {0}, already exists".format(column))
            if loc > len(self.columns):
                raise IndexError(
                    "index {0} is out of bounds for axis 0 with size {1}".format(
                        loc, len(self.columns)
                    )
                )
            if loc < 0:
                raise ValueError("unbounded slice")
            new_query_compiler = self._query_compiler.insert(loc, column, value)

        self._update_inplace(new_query_compiler=new_query_compiler)

    def interpolate(
        self,
        method="linear",
        axis=0,
        limit=None,
        inplace=False,
        limit_direction="forward",
        limit_area=None,
        downcast=None,
        **kwargs
    ):
        return self._default_to_pandas(
            pandas.DataFrame.interpolate,
            method=method,
            axis=axis,
            limit=limit,
            inplace=inplace,
            limit_direction=limit_direction,
            limit_area=limit_area,
            downcast=downcast,
            **kwargs
        )

    def iterrows(self):
        """Iterate over DataFrame rows as (index, Series) pairs.

        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A generator that iterates over the rows of the frame.
        """
        index_iter = iter(self.index)

        def iterrow_builder(df):
            df.columns = self.columns
            df.index = [next(index_iter)]
            return df.iterrows()

        partition_iterator = PartitionIterator(self._query_compiler, 0, iterrow_builder)
        for v in partition_iterator:
            yield v

    def items(self):
        """Iterator over (column name, Series) pairs.

        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A generator that iterates over the columns of the frame.
        """
        col_iter = iter(self.columns)

        def items_builder(df):
            df.columns = [next(col_iter)]
            df.index = self.index
            return df.items()

        partition_iterator = PartitionIterator(self._query_compiler, 1, items_builder)
        for v in partition_iterator:
            yield v

    def iteritems(self):
        """Iterator over (column name, Series) pairs.

        Note:
            Returns the same thing as .items()

        Returns:
            A generator that iterates over the columns of the frame.
        """
        return self.items()

    def itertuples(self, index=True, name="Pandas"):
        """Iterate over DataFrame rows as namedtuples.

        Args:
            index (boolean, default True): If True, return the index as the
                first element of the tuple.
            name (string, default "Pandas"): The name of the returned
            namedtuples or None to return regular tuples.
        Note:
            Generators can't be pickled so from the remote function
            we expand the generator into a list before getting it.
            This is not that ideal.

        Returns:
            A tuple representing row data. See args for varying tuples.
        """
        index_iter = iter(self.index)

        def itertuples_builder(df):
            df.columns = self.columns
            df.index = [next(index_iter)]
            return df.itertuples(index=index, name=name)

        partition_iterator = PartitionIterator(
            self._query_compiler, 0, itertuples_builder
        )
        for v in partition_iterator:
            yield v

    def join(self, other, on=None, how="left", lsuffix="", rsuffix="", sort=False):
        """Join two or more DataFrames, or a DataFrame with a collection.

        Args:
            other: What to join this DataFrame with.
            on: A column name to use from the left for the join.
            how: What type of join to conduct.
            lsuffix: The suffix to add to column names that match on left.
            rsuffix: The suffix to add to column names that match on right.
            sort: Whether or not to sort.

        Returns:
            The joined DataFrame.
        """

        if on is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.join,
                other,
                on=on,
                how=how,
                lsuffix=lsuffix,
                rsuffix=rsuffix,
                sort=sort,
            )
        if isinstance(other, pandas.Series):
            if other.name is None:
                raise ValueError("Other Series must have a name")
            other = DataFrame({other.name: other})
        if isinstance(other, DataFrame):
            # Joining the empty DataFrames with either index or columns is
            # fast. It gives us proper error checking for the edge cases that
            # would otherwise require a lot more logic.
            pandas.DataFrame(columns=self.columns).join(
                pandas.DataFrame(columns=other.columns),
                lsuffix=lsuffix,
                rsuffix=rsuffix,
            ).columns

            return DataFrame(
                query_compiler=self._query_compiler.join(
                    other._query_compiler,
                    how=how,
                    lsuffix=lsuffix,
                    rsuffix=rsuffix,
                    sort=sort,
                )
            )
        else:
            # This constraint carried over from Pandas.
            if on is not None:
                raise ValueError(
                    "Joining multiple DataFrames only supported for joining on index"
                )
            # See note above about error checking with an empty join.
            pandas.DataFrame(columns=self.columns).join(
                [pandas.DataFrame(columns=obj.columns) for obj in other],
                lsuffix=lsuffix,
                rsuffix=rsuffix,
            ).columns

            return DataFrame(
                query_compiler=self._query_compiler.join(
                    [obj._query_compiler for obj in other],
                    how=how,
                    lsuffix=lsuffix,
                    rsuffix=rsuffix,
                    sort=sort,
                )
            )

    def le(self, other, axis="columns", level=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).le(other, axis=axis, level=level)

    def lookup(self, row_labels, col_labels):
        return self._default_to_pandas(pandas.DataFrame.lookup, row_labels, col_labels)

    def lt(self, other, axis="columns", level=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).lt(other, axis=axis, level=level)

    def melt(
        self,
        id_vars=None,
        value_vars=None,
        var_name=None,
        value_name="value",
        col_level=None,
    ):
        return self._default_to_pandas(
            pandas.DataFrame.melt,
            id_vars=id_vars,
            value_vars=value_vars,
            var_name=var_name,
            value_name=value_name,
            col_level=col_level,
        )

    def memory_usage(self, index=True, deep=False):
        """Returns the memory usage of each column in bytes

        Args:
            index (bool): Whether to include the memory usage of the DataFrame's
                index in returned Series. Defaults to True
            deep (bool): If True, introspect the data deeply by interrogating
            objects dtypes for system-level memory consumption. Defaults to False

        Returns:
            A Series where the index are the column names and the values are
            the memory usage of each of the columns in bytes. If `index=true`,
            then the first value of the Series will be 'Index' with its memory usage.
        """
        if index:
            result = self._reduce_dimension(
                self._query_compiler.memory_usage(index=False, deep=deep)
            )
            index_value = self.index.memory_usage(deep=deep)
            return Series(index_value, index=["Index"]).append(result)
        return super(DataFrame, self).memory_usage(index=index, deep=deep)

    def merge(
        self,
        right,
        how="inner",
        on=None,
        left_on=None,
        right_on=None,
        left_index=False,
        right_index=False,
        sort=False,
        suffixes=("_x", "_y"),
        copy=True,
        indicator=False,
        validate=None,
    ):
        """Database style join, where common columns in "on" are merged.

        Args:
            right: The DataFrame to merge against.
            how: What type of join to use.
            on: The common column name(s) to join on. If None, and left_on and
                right_on  are also None, will default to all commonly named
                columns.
            left_on: The column(s) on the left to use for the join.
            right_on: The column(s) on the right to use for the join.
            left_index: Use the index from the left as the join keys.
            right_index: Use the index from the right as the join keys.
            sort: Sort the join keys lexicographically in the result.
            suffixes: Add this suffix to the common names not in the "on".
            copy: Does nothing in our implementation
            indicator: Adds a column named _merge to the DataFrame with
                metadata from the merge about each row.
            validate: Checks if merge is a specific type.

        Returns:
             A merged Dataframe
        """

        if not isinstance(right, DataFrame):
            raise ValueError(
                "can not merge DataFrame with instance of type "
                "{}".format(type(right))
            )
        if left_index is False or right_index is False:
            if isinstance(right, DataFrame):
                right = right._query_compiler.to_pandas()
            return self._default_to_pandas(
                pandas.DataFrame.merge,
                right,
                how=how,
                on=on,
                left_on=left_on,
                right_on=right_on,
                left_index=left_index,
                right_index=right_index,
                sort=sort,
                suffixes=suffixes,
                copy=copy,
                indicator=indicator,
                validate=validate,
            )
        if left_index and right_index:
            return self.join(
                right, how=how, lsuffix=suffixes[0], rsuffix=suffixes[1], sort=sort
            )

    def mod(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).mod(
            other, axis=axis, level=level, fill_value=None
        )

    def mul(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).mul(
            other, axis=axis, level=level, fill_value=None
        )

    rmul = multiply = mul

    def ne(self, other, axis="columns", level=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).ne(other, axis=axis, level=level)

    def nlargest(self, n, columns, keep="first"):
        return self._default_to_pandas(pandas.DataFrame.nlargest, n, columns, keep=keep)

    def nsmallest(self, n, columns, keep="first"):
        return self._default_to_pandas(
            pandas.DataFrame.nsmallest, n, columns, keep=keep
        )

    def pivot(self, index=None, columns=None, values=None):
        return self._default_to_pandas(
            pandas.DataFrame.pivot, index=index, columns=columns, values=values
        )

    def pivot_table(
        self,
        values=None,
        index=None,
        columns=None,
        aggfunc="mean",
        fill_value=None,
        margins=False,
        dropna=True,
        margins_name="All",
    ):
        return self._default_to_pandas(
            pandas.DataFrame.pivot_table,
            values=values,
            index=index,
            columns=columns,
            aggfunc=aggfunc,
            fill_value=fill_value,
            margins=margins,
            dropna=dropna,
            margins_name=margins_name,
        )

    @property
    def plot(
        self,
        x=None,
        y=None,
        kind="line",
        ax=None,
        subplots=False,
        sharex=None,
        sharey=False,
        layout=None,
        figsize=None,
        use_index=True,
        title=None,
        grid=None,
        legend=True,
        style=None,
        logx=False,
        logy=False,
        loglog=False,
        xticks=None,
        yticks=None,
        xlim=None,
        ylim=None,
        rot=None,
        fontsize=None,
        colormap=None,
        table=False,
        yerr=None,
        xerr=None,
        secondary_y=False,
        sort_columns=False,
        **kwargs
    ):
        return self._to_pandas().plot

    def pow(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).pow(
            other, axis=axis, level=level, fill_value=None
        )

    def prod(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs
    ):
        axis = self._get_axis_number(axis)
        new_index = self.columns if axis else self.index
        if min_count > len(new_index):
            return Series(
                [np.nan] * len(new_index), index=new_index, dtype=np.dtype("object")
            )
        return super(DataFrame, self).prod(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs
        )

    product = prod
    radd = add

    def query(self, expr, inplace=False, **kwargs):
        """Queries the Dataframe with a boolean expression

        Returns:
            A new DataFrame if inplace=False
        """
        ErrorMessage.non_verified_udf()
        self._validate_eval_query(expr, **kwargs)
        inplace = validate_bool_kwarg(inplace, "inplace")
        new_query_compiler = self._query_compiler.query(expr, **kwargs)
        return self._create_or_update_from_compiler(new_query_compiler, inplace)

    def rename(
        self,
        mapper=None,
        index=None,
        columns=None,
        axis=None,
        copy=True,
        inplace=False,
        level=None,
    ):
        """Alters axes labels.

        Args:
            mapper, index, columns: Transformations to apply to the axis's
                values.
            axis: Axis to target with mapper.
            copy: Also copy underlying data.
            inplace: Whether to return a new DataFrame.
            level: Only rename a specific level of a MultiIndex.

        Returns:
            If inplace is False, a new DataFrame with the updated axes.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        # We have to do this with the args because of how rename handles kwargs. It
        # doesn't ignore None values passed in, so we have to filter them ourselves.
        args = locals()
        kwargs = {k: v for k, v in args.items() if v is not None and k != "self"}
        # inplace should always be true because this is just a copy, and we will use the
        # results after.
        kwargs["inplace"] = False
        if index is not None:
            new_index = pandas.DataFrame(index=self.index).rename(**kwargs).index
        else:
            new_index = self.index
        if columns is not None:
            new_columns = (
                pandas.DataFrame(columns=self.columns).rename(**kwargs).columns
            )
        else:
            new_columns = self.columns

        if inplace:
            obj = self
        else:
            obj = self.copy()
        obj.index = new_index
        obj.columns = new_columns

        if not inplace:
            return obj

    def _set_axis_name(self, name, axis=0, inplace=False):
        """Alter the name or names of the axis.

        Args:
            name: Name for the Index, or list of names for the MultiIndex
            axis: 0 or 'index' for the index; 1 or 'columns' for the columns
            inplace: Whether to modify `self` directly or return a copy

        Returns:
            Type of caller or None if inplace=True.
        """
        axis = self._get_axis_number(axis) if axis is not None else 0
        renamed = self if inplace else self.copy()
        if axis == 0:
            renamed.index = renamed.index.set_names(name)
        else:
            renamed.columns = renamed.columns.set_names(name)
        if not inplace:
            return renamed

    def reorder_levels(self, order, axis=0):
        return self._default_to_pandas(
            pandas.DataFrame.reorder_levels, order, axis=axis
        )

    def rfloordiv(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).rfloordiv(
            other, axis=axis, level=level, fill_value=None
        )

    def rmod(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).rmod(
            other, axis=axis, level=level, fill_value=None
        )

    def rpow(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).rpow(
            other, axis=axis, level=level, fill_value=None
        )

    def rsub(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).rsub(
            other, axis=axis, level=level, fill_value=None
        )

    def rtruediv(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).rtruediv(
            other, axis=axis, level=level, fill_value=None
        )

    rdiv = rtruediv

    def select_dtypes(self, include=None, exclude=None):
        # Validates arguments for whether both include and exclude are None or
        # if they are disjoint. Also invalidates string dtypes.
        pandas.DataFrame().select_dtypes(include, exclude)

        if include and not is_list_like(include):
            include = [include]
        elif include is None:
            include = []
        if exclude and not is_list_like(exclude):
            exclude = [exclude]
        elif exclude is None:
            exclude = []

        sel = tuple(map(set, (include, exclude)))
        include, exclude = map(lambda x: set(map(infer_dtype_from_object, x)), sel)
        include_these = pandas.Series(not bool(include), index=self.columns)
        exclude_these = pandas.Series(not bool(exclude), index=self.columns)

        def is_dtype_instance_mapper(column, dtype):
            return column, functools.partial(issubclass, dtype.type)

        for column, f in itertools.starmap(
            is_dtype_instance_mapper, self.dtypes.iteritems()
        ):
            if include:  # checks for the case of empty include or exclude
                include_these[column] = any(map(f, include))
            if exclude:
                exclude_these[column] = not any(map(f, exclude))

        dtype_indexer = include_these & exclude_these
        indicate = [
            i for i in range(len(dtype_indexer.values)) if not dtype_indexer.values[i]
        ]
        return self.drop(columns=self.columns[indicate], inplace=False)

    def set_index(
        self, keys, drop=True, append=False, inplace=False, verify_integrity=False
    ):
        """Set the DataFrame index using one or more existing columns.

        Args:
            keys: column label or list of column labels / arrays.
            drop (boolean): Delete columns to be used as the new index.
            append (boolean): Whether to append columns to existing index.
            inplace (boolean): Modify the DataFrame in place.
            verify_integrity (boolean): Check the new index for duplicates.
                Otherwise defer the check until necessary. Setting to False
                will improve the performance of this method

        Returns:
            If inplace is set to false returns a new DataFrame, otherwise None.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if not isinstance(keys, list):
            keys = [keys]
        if inplace:
            frame = self
        else:
            frame = self.copy()

        arrays = []
        names = []
        if append:
            names = [x for x in self.index.names]
            if isinstance(self.index, pandas.MultiIndex):
                for i in range(self.index.nlevels):
                    arrays.append(self.index._get_level_values(i))
            else:
                arrays.append(self.index)
        to_remove = []
        for col in keys:
            if isinstance(col, pandas.MultiIndex):
                # append all but the last column so we don't have to modify
                # the end of this loop
                for n in range(col.nlevels - 1):
                    arrays.append(col._get_level_values(n))

                level = col._get_level_values(col.nlevels - 1)
                names.extend(col.names)
            elif isinstance(col, pandas.Series):
                level = col._values
                names.append(col.name)
            elif isinstance(col, pandas.Index):
                level = col
                names.append(col.name)
            elif isinstance(col, (list, np.ndarray, pandas.Index)):
                level = col
                names.append(None)
            else:
                level = frame[col]._to_pandas()._values
                names.append(col)
                if drop:
                    to_remove.append(col)
            arrays.append(level)
        index = ensure_index_from_sequences(arrays, names)

        if verify_integrity and not index.is_unique:
            duplicates = index.get_duplicates()
            raise ValueError("Index has duplicate keys: %s" % duplicates)

        for c in to_remove:
            del frame[c]
        # clear up memory usage
        index._cleanup()
        frame.index = index

        if not inplace:
            return frame

    def squeeze(self, axis=None):
        axis = self._get_axis_number(axis) if axis is not None else None
        if axis is None and (len(self.columns) == 1 or len(self.index) == 1):
            return Series(query_compiler=self._query_compiler).squeeze()
        if axis == 1 and len(self.columns) == 1:
            return Series(query_compiler=self._query_compiler)
        if axis == 0 and len(self.index) == 1:
            return Series(query_compiler=self._query_compiler)
        else:
            return self.copy()

    def stack(self, level=-1, dropna=True):
        return self._default_to_pandas(
            pandas.DataFrame.stack, level=level, dropna=dropna
        )

    def sub(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).sub(
            other, axis=axis, level=level, fill_value=None
        )

    subtract = sub

    def sum(
        self,
        axis=None,
        skipna=None,
        level=None,
        numeric_only=None,
        min_count=0,
        **kwargs
    ):
        axis = self._get_axis_number(axis)
        new_index = self.columns if axis else self.index
        if min_count > len(new_index):
            return Series(
                [np.nan] * len(new_index), index=new_index, dtype=np.dtype("object")
            )
        return super(DataFrame, self).sum(
            axis=axis,
            skipna=skipna,
            level=level,
            numeric_only=numeric_only,
            min_count=min_count,
            **kwargs
        )

    def tail(self, n=5):
        if n == 0:
            return DataFrame(columns=self.columns)
        return super(DataFrame, self).tail(n)

    def to_feather(self, fname):  # pragma: no cover
        return self._default_to_pandas(pandas.DataFrame.to_feather, fname)

    def to_gbq(
        self,
        destination_table,
        project_id=None,
        chunksize=None,
        reauth=False,
        if_exists="fail",
        auth_local_webserver=False,
        table_schema=None,
        location=None,
        progress_bar=True,
        credentials=None,
        verbose=None,
        private_key=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_gbq,
            destination_table,
            project_id=project_id,
            chunksize=chunksize,
            reauth=reauth,
            if_exists=if_exists,
            auth_local_webserver=auth_local_webserver,
            table_schema=table_schema,
            location=location,
            progress_bar=progress_bar,
            credentials=credentials,
            verbose=verbose,
            private_key=private_key,
        )

    def to_html(
        self,
        buf=None,
        columns=None,
        col_space=None,
        header=True,
        index=True,
        na_rep="NaN",
        formatters=None,
        float_format=None,
        sparsify=None,
        index_names=True,
        justify=None,
        max_rows=None,
        max_cols=None,
        show_dimensions=False,
        decimal=".",
        bold_rows=True,
        classes=None,
        escape=True,
        notebook=False,
        border=None,
        table_id=None,
        render_links=False,
    ):
        return self._default_to_pandas(
            pandas.DataFrame.to_html,
            buf=buf,
            columns=columns,
            col_space=col_space,
            header=header,
            index=index,
            na_rep=na_rep,
            formatters=formatters,
            float_format=float_format,
            sparsify=sparsify,
            index_names=index_names,
            justify=justify,
            max_rows=max_rows,
            max_cols=max_cols,
            show_dimensions=show_dimensions,
            decimal=decimal,
            bold_rows=bold_rows,
            classes=classes,
            escape=escape,
            notebook=notebook,
            border=border,
            table_id=table_id,
            render_links=render_links,
        )

    def to_panel(self):  # pragma: no cover
        return self._default_to_pandas(pandas.DataFrame.to_panel)

    def to_parquet(
        self,
        fname,
        engine="auto",
        compression="snappy",
        index=None,
        partition_cols=None,
        **kwargs
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_parquet,
            fname,
            engine=engine,
            compression=compression,
            index=index,
            partition_cols=partition_cols,
            **kwargs
        )

    def to_period(self, freq=None, axis=0, copy=True):  # pragma: no cover
        return super(DataFrame, self).to_period(freq=freq, axis=axis, copy=copy)

    def to_records(
        self, index=True, convert_datetime64=None, column_dtypes=None, index_dtypes=None
    ):
        return self._default_to_pandas(
            pandas.DataFrame.to_records,
            index=index,
            convert_datetime64=convert_datetime64,
            column_dtypes=column_dtypes,
            index_dtypes=index_dtypes,
        )

    def to_stata(
        self,
        fname,
        convert_dates=None,
        write_index=True,
        encoding="latin-1",
        byteorder=None,
        time_stamp=None,
        data_label=None,
        variable_labels=None,
        version=114,
        convert_strl=None,
    ):  # pragma: no cover
        return self._default_to_pandas(
            pandas.DataFrame.to_stata,
            fname,
            convert_dates=convert_dates,
            write_index=write_index,
            encoding=encoding,
            byteorder=byteorder,
            time_stamp=time_stamp,
            data_label=data_label,
            variable_labels=variable_labels,
            version=version,
            convert_strl=convert_strl,
        )

    def to_timestamp(self, freq=None, how="start", axis=0, copy=True):
        return super(DataFrame, self).to_timestamp(
            freq=freq, how=how, axis=axis, copy=copy
        )

    def truediv(self, other, axis="columns", level=None, fill_value=None):
        if isinstance(other, Series):
            other = other._to_pandas()
        return super(DataFrame, self).truediv(
            other, axis=axis, level=level, fill_value=None
        )

    div = divide = truediv

    def update(
        self, other, join="left", overwrite=True, filter_func=None, errors="ignore"
    ):
        """Modify DataFrame in place using non-NA values from other.

        Args:
            other: DataFrame, or object coercible into a DataFrame
            join: {'left'}, default 'left'
            overwrite: If True then overwrite values for common keys in frame
            filter_func: Can choose to replace values other than NA.
            raise_conflict: If True, will raise an error if the DataFrame and
                other both contain data in the same place.

        Returns:
            None
        """
        if errors == "raise":
            return self._default_to_pandas(
                pandas.DataFrame.update,
                other,
                join=join,
                overwrite=overwrite,
                filter_func=filter_func,
                errors=errors,
            )
        if not isinstance(other, DataFrame):
            other = DataFrame(other)
        query_compiler = self._query_compiler.update(
            other._query_compiler,
            join=join,
            overwrite=overwrite,
            filter_func=filter_func,
            errors=errors,
        )
        self._update_inplace(new_query_compiler=query_compiler)

    def where(
        self,
        cond,
        other=np.nan,
        inplace=False,
        axis=None,
        level=None,
        errors="raise",
        try_cast=False,
        raise_on_error=None,
    ):
        """Replaces values not meeting condition with values in other.

        Args:
            cond: A condition to be met, can be callable, array-like or a
                DataFrame.
            other: A value or DataFrame of values to use for setting this.
            inplace: Whether or not to operate inplace.
            axis: The axis to apply over. Only valid when a Series is passed
                as other.
            level: The MultiLevel index level to apply over.
            errors: Whether or not to raise errors. Does nothing in Pandas.
            try_cast: Try to cast the result back to the input type.
            raise_on_error: Whether to raise invalid datatypes (deprecated).

        Returns:
            A new DataFrame with the replaced values.
        """
        inplace = validate_bool_kwarg(inplace, "inplace")
        if isinstance(other, pandas.Series) and axis is None:
            raise ValueError("Must specify axis=0 or 1")
        if level is not None:
            if isinstance(other, DataFrame):
                other = other._query_compiler.to_pandas()
            if isinstance(cond, DataFrame):
                cond = cond._query_compiler.to_pandas()
            new_query_compiler = self._default_to_pandas(
                pandas.DataFrame.where,
                cond,
                other=other,
                inplace=False,
                axis=axis,
                level=level,
                errors=errors,
                try_cast=try_cast,
                raise_on_error=raise_on_error,
            )
            return self._create_or_update_from_compiler(new_query_compiler, inplace)
        axis = self._get_axis_number(axis) if axis is not None else 0
        cond = cond(self) if callable(cond) else cond

        if not isinstance(cond, DataFrame):
            if not hasattr(cond, "shape"):
                cond = np.asanyarray(cond)
            if cond.shape != self.shape:
                raise ValueError("Array conditional must be same shape as self")
            cond = DataFrame(cond, index=self.index, columns=self.columns)
        if isinstance(other, DataFrame):
            other = other._query_compiler
        elif isinstance(other, pandas.Series):
            other = other.reindex(self.index if not axis else self.columns)
        else:
            index = self.index if not axis else self.columns
            other = pandas.Series(other, index=index)
        query_compiler = self._query_compiler.where(
            cond._query_compiler, other, axis=axis, level=level
        )
        return self._create_or_update_from_compiler(query_compiler, inplace)

    def xs(self, key, axis=0, level=None, drop_level=True):
        return self._default_to_pandas(
            pandas.DataFrame.xs, key, axis=axis, level=level, drop_level=drop_level
        )

    def _getitem(self, key):
        """Get the column specified by key for this DataFrame.

        Args:
            key : The column name.

        Returns:
            A Pandas Series representing the value for the column.
        """
        key = apply_if_callable(key, self)
        # Shortcut if key is an actual column
        is_mi_columns = isinstance(self.columns, pandas.MultiIndex)
        try:
            if key in self.columns and not is_mi_columns:
                return self._getitem_column(key)
        except (KeyError, ValueError, TypeError):
            pass
        if isinstance(key, (Series, np.ndarray, pandas.Index, list)):
            return self._getitem_array(key)
        elif isinstance(key, DataFrame):
            return self.where(key)
        elif is_mi_columns:
            return self._default_to_pandas(pandas.DataFrame.__getitem__, key)
            # return self._getitem_multilevel(key)
        else:
            return self._getitem_column(key)

    def _getitem_column(self, key):
        if key not in self.keys():
            raise KeyError("{}".format(key))
        s = self._reduce_dimension(self._query_compiler.getitem_column_array([key]))
        s._parent = self
        return s

    def _getitem_array(self, key):
        # TODO: dont convert to pandas for array indexing
        if isinstance(key, Series):
            key = key._to_pandas()
        if is_bool_indexer(key):
            if isinstance(key, pandas.Series) and not key.index.equals(self.index):
                warnings.warn(
                    "Boolean Series key will be reindexed to match DataFrame index.",
                    PendingDeprecationWarning,
                    stacklevel=3,
                )
            elif len(key) != len(self.index):
                raise ValueError(
                    "Item wrong length {} instead of {}.".format(
                        len(key), len(self.index)
                    )
                )
            key = check_bool_indexer(self.index, key)
            # We convert to a RangeIndex because getitem_row_array is expecting a list
            # of indices, and RangeIndex will give us the exact indices of each boolean
            # requested.
            key = pandas.RangeIndex(len(self.index))[key]
            if len(key):
                return DataFrame(
                    query_compiler=self._query_compiler.getitem_row_array(key)
                )
            else:
                return DataFrame(columns=self.columns)
        else:
            if any(k not in self.columns for k in key):
                raise KeyError(
                    "{} not index".format(
                        str([k for k in key if k not in self.columns]).replace(",", "")
                    )
                )
            return DataFrame(
                query_compiler=self._query_compiler.getitem_column_array(key)
            )

    def __getattr__(self, key):
        """After regular attribute access, looks up the name in the columns

        Args:
            key (str): Attribute name.

        Returns:
            The value of the attribute.
        """
        try:
            return object.__getattribute__(self, key)
        except AttributeError as e:
            if key in self.columns:
                return self[key]
            raise e

    def __setattr__(self, key, value):
        # We have to check for this first because we have to be able to set
        # _query_compiler before we check if the key is in self
        if key in ["_query_compiler"] or key in self.__dict__:
            pass
        elif key in self:
            self.__setitem__(key, value)
        elif isinstance(value, pandas.Series):
            warnings.warn(
                "Modin doesn't allow columns to be created via a new attribute name - see "
                "https://pandas.pydata.org/pandas-docs/stable/indexing.html#attribute-access",
                UserWarning,
            )
        object.__setattr__(self, key, value)

    def __setitem__(self, key, value):
        if not isinstance(key, str):

            def setitem_without_string_columns(df):
                # Arrow makes memory-mapped objects immutable, so copy will allow them
                # to be mutable again.
                df = df.copy(True)
                df[key] = value
                return df

            return self._update_inplace(
                self._default_to_pandas(setitem_without_string_columns)._query_compiler
            )
        if is_list_like(value):
            if isinstance(value, (pandas.DataFrame, DataFrame)):
                if value.shape[1] != 1 and key not in self.columns:
                    raise ValueError(
                        "Wrong number of items passed %i, placement implies 1"
                        % value.shape[1]
                    )
                value = value[value.columns[0]].values
            elif isinstance(value, np.ndarray):
                if (
                    len(value.shape) > 1
                    and value.shape[1] != 1
                    and key not in self.columns
                ):
                    raise ValueError(
                        "Wrong number of items passed %i, placement implies 1"
                        % value.shape[1]
                    )
                assert (
                    len(value.shape) < 3
                ), "Shape of new values must be compatible with manager shape"
                value = value.T.reshape(-1)[: len(self)]
            if not isinstance(value, Series):
                value = list(value)
        if key not in self.columns:
            if isinstance(value, Series):
                self._create_or_update_from_compiler(
                    self._query_compiler.concat(1, value._query_compiler), inplace=True
                )
                # Now that the data is appended, we need to update the column name for
                # that column to `key`, otherwise the name could be incorrect. Drop the
                # last column name from the list (the appended value's name and append
                # the new name.
                self.columns = self.columns[:-1].append(pandas.Index([key]))
            else:
                self.insert(loc=len(self.columns), column=key, value=value)
        elif len(self.index) == 0:
            new_self = DataFrame({key: value}, columns=self.columns)
            self._update_inplace(new_self._query_compiler)
        else:
            self._update_inplace(self._query_compiler.setitem(0, key, value))

    def __unicode__(self):
        return self._default_to_pandas(pandas.DataFrame.__unicode__)

    def __hash__(self):
        return self._default_to_pandas(pandas.DataFrame.__hash__)

    def __iter__(self):
        """Iterate over the columns

        Returns:
            An Iterator over the columns of the DataFrame.
        """
        return iter(self.columns)

    def __contains__(self, key):
        """Searches columns for specific key

        Args:
            key : The column name

        Returns:
            Returns a boolean if the specified key exists as a column name
        """
        return self.columns.__contains__(key)

    def __round__(self, decimals=0):
        return self._default_to_pandas(pandas.DataFrame.__round__, decimals=decimals)

    def __setstate__(self, state):
        return self._default_to_pandas(pandas.DataFrame.__setstate__, state)

    def __delitem__(self, key):
        """Delete a column by key. `del a[key]` for example.
           Operation happens in place.

           Notes: This operation happen on row and column partition
                  simultaneously. No rebuild.
        Args:
            key: key to delete
        """
        if key not in self:
            raise KeyError(key)
        self._update_inplace(new_query_compiler=self._query_compiler.delitem(key))

    def __add__(self, other, axis=None, level=None, fill_value=None):
        return self.add(other, axis=axis, level=level, fill_value=fill_value)

    def __iadd__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.add(other, axis=axis, level=level, fill_value=fill_value)

    def __radd__(self, other, axis=None, level=None, fill_value=None):
        return self.radd(other, axis=axis, level=level, fill_value=fill_value)

    def __mul__(self, other, axis=None, level=None, fill_value=None):
        return self.mul(other, axis=axis, level=level, fill_value=fill_value)

    def __imul__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.mul(other, axis=axis, level=level, fill_value=fill_value)

    def __rmul__(self, other, axis=None, level=None, fill_value=None):
        return self.rmul(other, axis=axis, level=level, fill_value=fill_value)

    def __pow__(self, other, axis=None, level=None, fill_value=None):
        return self.pow(other, axis=axis, level=level, fill_value=fill_value)

    def __ipow__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.pow(other, axis=axis, level=level, fill_value=fill_value)

    def __rpow__(self, other, axis=None, level=None, fill_value=None):
        return self.rpow(other, axis=axis, level=level, fill_value=fill_value)

    def __sub__(self, other, axis=None, level=None, fill_value=None):
        return self.sub(other, axis=axis, level=level, fill_value=fill_value)

    def __isub__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.sub(other, axis=axis, level=level, fill_value=fill_value)

    def __rsub__(self, other, axis=None, level=None, fill_value=None):
        return self.rsub(other, axis=axis, level=level, fill_value=fill_value)

    def __floordiv__(self, other, axis=None, level=None, fill_value=None):
        return self.floordiv(other, axis=axis, level=level, fill_value=fill_value)

    def __ifloordiv__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.floordiv(other, axis=axis, level=level, fill_value=fill_value)

    def __rfloordiv__(self, other, axis=None, level=None, fill_value=None):
        return self.rfloordiv(other, axis=axis, level=level, fill_value=fill_value)

    def __truediv__(self, other, axis=None, level=None, fill_value=None):
        return self.truediv(other, axis=axis, level=level, fill_value=fill_value)

    def __itruediv__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.truediv(other, axis=axis, level=level, fill_value=fill_value)

    def __rtruediv__(self, other, axis=None, level=None, fill_value=None):
        return self.rtruediv(other, axis=axis, level=level, fill_value=fill_value)

    def __mod__(self, other, axis=None, level=None, fill_value=None):
        return self.mod(other, axis=axis, level=level, fill_value=fill_value)

    def __imod__(
        self, other, axis=None, level=None, fill_value=None
    ):  # pragma: no cover
        return self.mod(other, axis=axis, level=level, fill_value=fill_value)

    def __rmod__(self, other, axis=None, level=None, fill_value=None):
        return self.rmod(other, axis=axis, level=level, fill_value=fill_value)

    def __div__(self, other, axis=None, level=None, fill_value=None):
        return self.div(other, axis=axis, level=level, fill_value=fill_value)

    def __rdiv__(self, other, axis=None, level=None, fill_value=None):
        return self.rdiv(other, axis=axis, level=level, fill_value=fill_value)

    @property
    def __doc__(self):  # pragma: no cover
        def __doc__(df):
            """Defined because properties do not have a __name__"""
            return df.__doc__

        return self._default_to_pandas(__doc__)

    @property
    def style(self):
        def style(df):
            """Defined because properties do not have a __name__"""
            return df.style

        return self._default_to_pandas(style)

    def _create_or_update_from_compiler(self, new_query_compiler, inplace=False):
        """Returns or updates a DataFrame given new query_compiler"""
        assert (
            isinstance(new_query_compiler, type(self._query_compiler))
            or type(new_query_compiler) in self._query_compiler.__class__.__bases__
        ), "Invalid Query Compiler object: {}".format(type(new_query_compiler))
        if not inplace:
            return DataFrame(query_compiler=new_query_compiler)
        else:
            self._update_inplace(new_query_compiler=new_query_compiler)

    def _validate_dtypes(self, numeric_only=False):
        """Helper method to check that all the dtypes are the same"""
        dtype = self.dtypes[0]
        for t in self.dtypes:
            if numeric_only and not is_numeric_dtype(t):
                raise TypeError("{0} is not a numeric data type".format(t))
            elif not numeric_only and t != dtype:
                raise TypeError(
                    "Cannot compare type '{0}' with type '{1}'".format(t, dtype)
                )

    def _validate_dtypes_min_max(self, axis, numeric_only):
        # If our DataFrame has both numeric and non-numeric dtypes then
        # comparisons between these types do not make sense and we must raise a
        # TypeError. The exception to this rule is when there are datetime and
        # timedelta objects, in which case we proceed with the comparison
        # without ignoring any non-numeric types. We must check explicitly if
        # numeric_only is False because if it is None, it will default to True
        # if the operation fails with mixed dtypes.
        if (
            axis
            and numeric_only is False
            and np.unique([is_numeric_dtype(dtype) for dtype in self.dtypes]).size == 2
        ):
            # check if there are columns with dtypes datetime or timedelta
            if all(
                dtype != np.dtype("datetime64[ns]")
                and dtype != np.dtype("timedelta64[ns]")
                for dtype in self.dtypes
            ):
                raise TypeError("Cannot compare Numeric and Non-Numeric Types")
        # Pandas ignores `numeric_only` if `axis` is 1, but we do have to drop
        # non-numeric columns if `axis` is 0.
        if numeric_only and axis == 0:
            return self.drop(
                columns=[
                    i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
                ]
            )
        else:
            return self

    def _validate_dtypes_sum_prod_mean(self, axis, numeric_only, ignore_axis=False):
        """Raises TypeErrors for sum, prod, and mean where necessary"""
        # We cannot add datetime types, so if we are summing a column with
        # dtype datetime64 and cannot ignore non-numeric types, we must throw a
        # TypeError.
        if (
            not axis
            and numeric_only is False
            and any(dtype == np.dtype("datetime64[ns]") for dtype in self.dtypes)
        ):
            raise TypeError("Cannot add Timestamp Types")

        # If our DataFrame has both numeric and non-numeric dtypes then
        # operations between these types do not make sense and we must raise a
        # TypeError. The exception to this rule is when there are datetime and
        # timedelta objects, in which case we proceed with the comparison
        # without ignoring any non-numeric types. We must check explicitly if
        # numeric_only is False because if it is None, it will default to True
        # if the operation fails with mixed dtypes.
        if (
            (axis or ignore_axis)
            and numeric_only is False
            and np.unique([is_numeric_dtype(dtype) for dtype in self.dtypes]).size == 2
        ):
            # check if there are columns with dtypes datetime or timedelta
            if all(
                dtype != np.dtype("datetime64[ns]")
                and dtype != np.dtype("timedelta64[ns]")
                for dtype in self.dtypes
            ):
                raise TypeError("Cannot operate on Numeric and Non-Numeric Types")
        # Pandas ignores `numeric_only` if `axis` is 1, but we do have to drop
        # non-numeric columns if `axis` is 0.
        if numeric_only and axis == 0:
            return self.drop(
                columns=[
                    i for i in self.dtypes.index if not is_numeric_dtype(self.dtypes[i])
                ]
            )
        else:
            return self

    def _to_pandas(self):
        return self._query_compiler.to_pandas()
