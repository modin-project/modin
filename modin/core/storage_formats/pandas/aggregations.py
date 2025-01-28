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

"""Contains implementations for aggregation functions."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Callable, Tuple

import numpy as np
import pandas
from pandas.core.dtypes.common import is_numeric_dtype

if TYPE_CHECKING:
    from .query_compiler import PandasQueryCompiler

from modin.utils import MODIN_UNNAMED_SERIES_LABEL


class CorrCovBuilder:
    """Responsible for building pandas query compiler's methods computing correlation and covariance matrices."""

    class Method(Enum):
        """Enum specifying what method to use (either CORR for correlation or COV for covariance)."""

        CORR = 1
        COV = 2

    @classmethod
    def build_corr_method(
        cls,
    ) -> Callable[[PandasQueryCompiler, str, int, bool], PandasQueryCompiler]:
        """
        Build a query compiler method computing the correlation matrix.

        Returns
        -------
        callable(qc: PandasQueryCompiler, method: str, min_periods: int, numeric_only: bool) -> PandasQueryCompiler
            A callable matching the ``BaseQueryCompiler.corr`` signature and computing the correlation matrix.
        """

        def corr_method(
            qc: PandasQueryCompiler,
            method: str,
            min_periods: int = 1,
            numeric_only: bool = True,
        ) -> PandasQueryCompiler:
            # Further implementation is designed for the default pandas backend (numpy)
            if method != "pearson" or qc.get_pandas_backend() == "pyarrow":
                return super(type(qc), qc).corr(
                    method=method, min_periods=min_periods, numeric_only=numeric_only
                )

            if not numeric_only and qc.frame_has_materialized_columns:
                new_index, new_columns = (
                    qc._modin_frame.copy_columns_cache(),
                    qc._modin_frame.copy_columns_cache(),
                )
                new_dtypes = pandas.Series(
                    np.repeat(pandas.api.types.pandas_dtype("float"), len(new_columns)),
                    index=new_columns,
                )
            elif numeric_only and qc.frame_has_materialized_dtypes:
                old_dtypes = qc.dtypes

                new_columns = old_dtypes[old_dtypes.map(is_numeric_dtype)].index
                new_index = new_columns.copy()
                new_dtypes = pandas.Series(
                    np.repeat(pandas.api.types.pandas_dtype("float"), len(new_columns)),
                    index=new_columns,
                )
            else:
                new_index, new_columns, new_dtypes = None, None, None

            map, reduce = cls._build_map_reduce_methods(
                min_periods, method=cls.Method.CORR, numeric_only=numeric_only
            )

            reduced = qc._modin_frame.apply_full_axis(axis=1, func=map)
            # The 'reduced' dataset has the shape either (num_cols, num_cols + 3) for a non-NaN case
            # or (num_cols, num_cols * 4) for a NaN case, so it's acceptable to call `.combine_and_apply()`
            # here as the number of cols is usually quite small
            result = reduced.combine_and_apply(
                func=reduce,
                new_index=new_index,
                new_columns=new_columns,
                new_dtypes=new_dtypes,
            )
            return qc.__constructor__(result)

        return corr_method

    @classmethod
    def build_cov_method(
        cls,
    ) -> Callable[[PandasQueryCompiler, int, int], PandasQueryCompiler]:
        """
        Build a query compiler method computing the covariance matrix.

        Returns
        -------
        callable(qc: PandasQueryCompiler, min_periods: int, ddof: int) -> PandasQueryCompiler
            A callable matching the ``BaseQueryCompiler.cov`` signature and computing the covariance matrix.
        """
        raise NotImplementedError("Computing covariance is not yet implemented.")

    @classmethod
    def _build_map_reduce_methods(
        cls, min_periods: int, method: Method, numeric_only: bool
    ) -> Tuple[
        Callable[[pandas.DataFrame], pandas.DataFrame],
        Callable[[pandas.DataFrame], pandas.DataFrame],
    ]:
        """
        Build MapReduce kernels for the specified corr/cov method.

        Parameters
        ----------
        min_periods : int
            The parameter to pass to the reduce method.
        method : CorrCovBuilder.Method
            Whether the kernels compute correlation or covariance.
        numeric_only : bool
            Whether to only include numeric types.

        Returns
        -------
        Tuple[Callable(pandas.DataFrame) -> pandas.DataFrame, Callable(pandas.DataFrame) -> pandas.DataFrame]
            A tuple holding the Map (at the first position) and the Reduce (at the second position) kernels
            computing correlation/covariance matrix.
        """
        if method == cls.Method.COV:
            raise NotImplementedError("Computing covariance is not yet implemented.")

        return lambda df: _CorrCovKernels.map(
            df, numeric_only
        ), lambda df: _CorrCovKernels.reduce(df, min_periods, method)


class _CorrCovKernels:
    """Holds kernel functions computing correlation/covariance matrices in a MapReduce manner."""

    @classmethod
    def map(cls, df: pandas.DataFrame, numeric_only: bool) -> pandas.DataFrame:
        """
        Perform the Map phase to compute the corr/cov matrix.

        In this kernel we compute all the required components to compute
        the correlation matrix at the reduce phase, the required components are:
            1. Matrix holding sums of pairwise multiplications between all columns
               defined as ``M[col1, col2] = sum(col1[i] * col2[i] for i in range(col_len))``
            2. Sum for each column (special case if there are NaN values)
            3. Sum of squares for each column (special case if there are NaN values)
            4. Number of values in each column (special case if there are NaN values)

        Parameters
        ----------
        df : pandas.DataFrame
            Partition to compute the aggregations for.
        numeric_only : bool
            Whether to only include numeric types.

        Returns
        -------
        pandas.DataFrame
            A MultiIndex columned DataFrame holding the described aggregation results for this
            specifix partition under the following keys: ``["mul", "sum", "pow2_sum", "count"]``
        """
        if numeric_only:
            df = df.select_dtypes(include="number")
        # It's more convenient to use a NumPy array here as it appears to perform
        # much faster in for-loops which this kernel function has plenty of
        raw_df = df.values.T
        try:
            nan_mask = np.isnan(raw_df)
        except TypeError as e:
            # Pandas raises ValueError on unsupported types, so casting
            # the exception to a proper type
            raise ValueError("Unsupported types with 'numeric_only=False'") from e

        has_nans = nan_mask.sum() != 0

        if has_nans:
            if not raw_df.flags.writeable:
                # making a copy if the buffer is read-only
                raw_df = raw_df.copy()
            # Replacing all NaNs with zeros so we can use much
            # faster `np.sum()` instead of slow `np.nansum()`
            np.putmask(raw_df, nan_mask, values=0)

        cols = df.columns
        # Here we compute a sum of pairwise multiplications between all columns
        # result:
        #   col1: [sum(col1 * col2), sum(col1 * col3), ... sum(col1 * colN)]
        #   col2: [sum(col2 * col3), sum(col2 * col4), ... sum(col2 * colN)]
        #   ...
        sum_of_pairwise_mul = pandas.DataFrame(
            np.dot(raw_df, raw_df.T), index=cols, columns=cols, copy=False
        )

        if has_nans:
            sums, sums_of_squares, count = cls._compute_nan_aggs(raw_df, cols, nan_mask)
        else:
            sums, sums_of_squares, count = cls._compute_non_nan_aggs(df)

        aggregations = pandas.concat(
            [sum_of_pairwise_mul, sums, sums_of_squares, count],
            copy=False,
            axis=1,
            keys=["mul", "sum", "pow2_sum", "count"],
        )

        return aggregations

    @staticmethod
    def _compute_non_nan_aggs(
        df: pandas.DataFrame,
    ) -> Tuple[pandas.Series, pandas.Series, pandas.Series]:
        """
        Compute sums, sums of square and the number of observations for a partition assuming there are no NaN values in it.

        Parameters
        ----------
        df : pandas.DataFrame
            Partition to compute the aggregations for.

        Returns
        -------
        Tuple[sums: pandas.Series, sums_of_squares: pandas.Series, count: pandas.Series]
            A tuple storing Series where each of them holds the result for
            one of the described aggregations.
        """
        sums = df.sum().rename(MODIN_UNNAMED_SERIES_LABEL)
        sums_of_squares = (df**2).sum().rename(MODIN_UNNAMED_SERIES_LABEL)
        count = pandas.Series(
            np.repeat(len(df), len(df.columns)), index=df.columns, copy=False
        ).rename(MODIN_UNNAMED_SERIES_LABEL)
        return sums, sums_of_squares, count

    @staticmethod
    def _compute_nan_aggs(
        raw_df: np.ndarray, cols: pandas.Index, nan_mask: np.ndarray
    ) -> Tuple[pandas.DataFrame, pandas.DataFrame, pandas.DataFrame]:
        """
        Compute sums, sums of square and the number of observations for a partition assuming there are NaN values in it.

        Parameters
        ----------
        raw_df : np.ndarray
            Raw values of the partition to compute the aggregations for.
        cols : pandas.Index
            Columns of the partition.
        nan_mask : np.ndarray[bool]
            Boolean mask showing positions of NaN values in the `raw_df`.

        Returns
        -------
        Tuple[sums: pandas.DataFrame, sums_of_squares: pandas.DataFrame, count: pandas.DataFrame]
            A tuple storing DataFrames where each of them holds the result for
            one of the described aggregations.
        """
        # Unfortunately, in case of NaN values we forced to compute multiple sums/square sums/counts
        # for each column because we have to exclude values at positions of NaN values in each other
        # column individually.
        # Imagine we have a dataframe like this:
        #   col1: 1, 2  , 3  , 4
        #   col2: 2, NaN, 3  , 4
        #   col3: 4, 5  , NaN, 7
        # In this case we would need to compute 2 different sums/square sums/count for 'col1':
        #   - The first one excluding the values at the NaN possitions of 'col2' (1 + 3 + 4)
        #   - And the second one excluding the values at the NaN positions of 'col3' (1 + 2 + 4)
        # and then also do the same for the rest columns. At the end this should form a matrix
        # of pairwise sums/square sums/counts:
        #   sums[col1, col2] = sum(col1[i] for i in non_NA_indices_of_col2)
        #   sums[col2, col1] = sum(col2[i] for i in non_NA_indices_of_col1)
        #   ...
        # Note that sums[col1, col2] != sums[col2, col1]
        sums = {}
        sums_of_squares = {}
        count = {}

        # TODO: is it possible to get rid of this for-loop somehow?
        for i, col in enumerate(cols):
            # Here we're taking each column, resizing it to the original frame's shape to compute
            # aggregations for each other column and then excluding values at those positions where
            # other columns had NaN values by setting zeros using the validity mask:
            #  col1: 1, 2  , 3  , 4   df[i].resize()  col1: 1, 2, 3, 4  putmask()  col1: 1, 2, 3, 4
            #  col2: 2, NaN, 3  , 4   ------------->  col1: 1, 2, 3, 4  -------->  col1: 1, 0, 3, 4
            #  col3: 4, 5  , NaN, 7                   col1: 1, 2, 3, 4             col1: 1, 2, 0, 4
            # Note that 'NaN' values in this diagram are just for the sake of visibility, in reality
            # they were already replaced by zeroes at the beginning of the 'map' phase.
            col_vals = np.resize(raw_df[i], raw_df.shape)
            np.putmask(col_vals, nan_mask, values=0)

            sums[col] = pandas.Series(np.sum(col_vals, axis=1), index=cols, copy=False)
            sums_of_squares[col] = pandas.Series(
                np.sum(col_vals**2, axis=1), index=cols, copy=False
            )
            count[col] = pandas.Series(
                nan_mask.shape[1] - np.count_nonzero(nan_mask | nan_mask[i], axis=1),
                index=cols,
                copy=False,
            )

        sums = pandas.concat(sums, axis=1, copy=False)
        sums_of_squares = pandas.concat(sums_of_squares, axis=1, copy=False)
        count = pandas.concat(count, axis=1, copy=False)

        return sums, sums_of_squares, count

    @classmethod
    def reduce(
        cls, df: pandas.DataFrame, min_periods: int, method: CorrCovBuilder.Method
    ) -> pandas.DataFrame:
        """
        Perform the Reduce phase to compute the corr/cov matrix.

        Parameters
        ----------
        df : pandas.DataFrame
            A dataframe holding aggregations computed for each partition
            concatenated along the rows axis.
        min_periods : int
            Minimum number of observations required per pair of columns to have a valid result.
        method : CorrCovBuilder.Method
            Whether to build a correlation or a covariance matrix.

        Returns
        -------
        pandas.DataFrame
            Either correlation or covariance matrix.
        """
        if method == CorrCovBuilder.Method.COV:
            raise NotImplementedError("Computing covariance is not yet implemented.")
        # The `df` here accumulates the aggregation results retrieved from each row partition
        # and combined together along the rows axis, so the `df` looks something like this:
        #   mul  sums  pow2_sums
        # a .    .     .
        # b .    .     .            <--- part1 result
        # c .    .     .
        # ---------------------------
        # a .    .     .
        # b .    .     .            <--- part2 result
        # c .    .     .
        # ---------------------------
        # ...
        # So to get the total result we have to group on the index and sum the values
        total_agg = df.groupby(level=0).sum()
        total_agg = cls._maybe_combine_nan_and_non_nan_aggs(total_agg)

        sum_of_pairwise_mul = total_agg["mul"]
        sums = total_agg["sum"]
        sums_of_squares = total_agg["pow2_sum"]
        count = total_agg["count"]

        cols = sum_of_pairwise_mul.columns
        # If there are NaNs in the original dataframe, then we have computed a matrix
        # of sums/square sums/counts at the Map phase, meaning that we now have multiple
        # columns in `sums`.
        has_nans = len(sums.columns) > 1
        if not has_nans:
            # 'count' is the same for all columns in a non-NaN case, so converting
            # it to scalar for faster binary operations
            count = count.iloc[0, 0]
            if count < min_periods:
                # Fast-path for too small data
                return pandas.DataFrame(index=cols, columns=cols, dtype="float")

            # Converting frame to a Series for more convenient handling
            sums = sums.squeeze(axis=1)
            sums_of_squares = sums_of_squares.squeeze(axis=1)

        means = sums / count
        std = np.sqrt(sums_of_squares - 2 * means * sums + count * (means**2))

        # The 'is_nans' condition was moved out of the loop, so the loops themselves
        # work faster as not being slowed by extra conditions in them
        if has_nans:
            return cls._build_corr_table_nan(
                sum_of_pairwise_mul, means, sums, count, std, cols, min_periods
            )
        else:
            # We've already processed the 'min_periods' parameter for a non-na case above,
            # so don't need to pass it here
            return cls._build_corr_table_non_nan(
                sum_of_pairwise_mul, means, sums, count, std, cols
            )

    @staticmethod
    def _maybe_combine_nan_and_non_nan_aggs(
        total_agg: pandas.DataFrame,
    ) -> pandas.DataFrame:
        """
        Pair the aggregation results of partitions having and not having NaN values if needed.

        Parameters
        ----------
        total_agg : pandas.DataFrame
            A dataframe holding aggregations computed for each partition
            concatenated along the rows axis.

        Returns
        -------
        pandas.DataFrame
            DataFrame with aligned results.
        """
        # Here we try to align the results between partitions that had and didn't have NaNs.
        # At the result of the Map phase, partitions with and without NaNs would produce
        # different results:
        #   - Partitions with NaNs produce a matrix of pairwise sums/square sums/counts
        #   - And parts without NaNs produce regular one-column sums/square sums/counts
        #
        # As the result, `total_agg` will be something like this:
        #    mul  | sum   pow2_sum  count | sum          pow2_sum     count
        #    a  b | a  b  a  b      a  b  | __reduced__  __reduced__  __reduced__
        # a  .  . | .  .  .  .      .  .  | .            .            .
        # b  .  . | .  .  .  .      .  .  | .            .            .
        # --------|-----------------------|----------------------------------------
        #           ^-- these are results   ^-- and these are results for
        #           for partitions that     partitions that didn't have NaNs
        #           had NaNs
        # So, to get an actual total result of these aggregations, we have to additionally
        # sum the results from non-NaN and NaN partitions.
        #
        # Here we sample the 'sum' columns to check whether we had mixed NaNs and
        # non-NaNs partitions, if it's not the case we can skip the described step:
        nsums = total_agg.columns.get_locs(["sum"])
        if not (
            len(nsums) > 1 and ("sum", MODIN_UNNAMED_SERIES_LABEL) in total_agg.columns
        ):
            return total_agg

        cols = total_agg.columns

        # Finding column positions for aggregational columns
        all_agg_idxs = np.where(
            cols.get_loc("sum") | cols.get_loc("pow2_sum") | cols.get_loc("count")
        )[0]
        # Finding column positions for aggregational columns that store
        # results of non-NaN partitions
        non_na_agg_idxs = cols.get_indexer_for(
            pandas.Index(
                [
                    ("sum", MODIN_UNNAMED_SERIES_LABEL),
                    ("pow2_sum", MODIN_UNNAMED_SERIES_LABEL),
                    ("count", MODIN_UNNAMED_SERIES_LABEL),
                ]
            )
        )
        # Finding column positions for aggregational columns that store
        # results of NaN partitions by deducting non-NaN indices from all indices
        na_agg_idxs = np.setdiff1d(all_agg_idxs, non_na_agg_idxs, assume_unique=True)

        # Using `.values` here so we can ignore the indices (it's really hard
        # to arrange them for pandas to properly perform the summation)
        parts_with_nans = total_agg.values[:, na_agg_idxs]
        parts_without_nans = (
            total_agg.values[:, non_na_agg_idxs]
            # Before doing the summation we have to align the shapes
            # Imagine that we have 'parts_with_nans' like:
            #    sum   pow2_sum  count
            #    a  b  a  b      a  b
            # a  1  2  3  4      5  6
            # b  1  2  3  4      5  6
            #
            # And the 'parts_without_nans' like:
            #    sum  pow2_sum  count
            # a  1    3         5
            # b  2    4         6
            #
            # Here we want to sum them in an order so the digit matches (1 + 1), (2 + 2), ...
            # For that we first have to repeat the values in 'parts_without_nans':
            #  parts_without_nans.repeat(parts_with_nans.shape[0]):
            #    sum  pow2_sum  count
            # a  1    3         5
            # b  1    3         5
            # a  2    4         6
            # b  2    4         6
            #
            # And then reshape it using the "Fortran" order:
            #  parts_without_nans.reshape(parts_with_nans.shape, order="F"):
            #    sum   pow2_sum  count
            #    a  b  a  b      a  b
            # a  1  2  3  4      5  6
            # b  1  2  3  4      5  6
            # After that the shapes & orders are aligned and we can perform the summation
            .repeat(repeats=len(parts_with_nans), axis=0).reshape(
                parts_with_nans.shape, order="F"
            )
        )
        replace_values = parts_with_nans + parts_without_nans

        if not total_agg.values.flags.writeable:
            # making a copy if the buffer is read-only as
            # we will need to modify `total_agg` inplace
            total_agg = total_agg.copy()
        total_agg.values[:, na_agg_idxs] = replace_values

        return total_agg

    @staticmethod
    def _build_corr_table_nan(
        sum_of_pairwise_mul: pandas.DataFrame,
        means: pandas.DataFrame,
        sums: pandas.DataFrame,
        count: pandas.DataFrame,
        std: pandas.DataFrame,
        cols: pandas.Index,
        min_periods: int,
    ) -> pandas.DataFrame:
        """
        Build correlation matrix for a DataFrame that had NaN values in it.

        Parameters
        ----------
        sum_of_pairwise_mul : pandas.DataFrame
        means : pandas.DataFrame
        sums : pandas.DataFrame
        count : pandas.DataFrame
        std : pandas.DataFrame
        cols : pandas.Index
        min_periods : int

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.
        """
        res = pandas.DataFrame(index=cols, columns=cols, dtype="float")
        nan_mask = count < min_periods

        for col in cols:
            top = (
                sum_of_pairwise_mul.loc[col]
                - sums.loc[col] * means[col]
                - means.loc[col] * sums[col]
                + count.loc[col] * means.loc[col] * means[col]
            )
            down = std.loc[col] * std[col]
            res.loc[col, :] = top / down

        res[nan_mask] = np.nan

        return res

    @staticmethod
    def _build_corr_table_non_nan(
        sum_of_pairwise_mul: pandas.DataFrame,
        means: pandas.Series,
        sums: pandas.Series,
        count: int,
        std: pandas.Series,
        cols: pandas.Index,
    ) -> pandas.DataFrame:
        """
        Build correlation matrix for a DataFrame that didn't have NaN values in it.

        Parameters
        ----------
        sum_of_pairwise_mul : pandas.DataFrame
        means : pandas.Series
        sums : pandas.Series
        count : int
        std : pandas.Series
        cols : pandas.Index

        Returns
        -------
        pandas.DataFrame
            Correlation matrix.
        """
        res = pandas.DataFrame(index=cols, columns=cols, dtype="float")

        for col in cols:
            top = (
                sum_of_pairwise_mul.loc[col]
                - sums.loc[col] * means
                - means.loc[col] * sums
                + count * means.loc[col] * means
            )
            down = std.loc[col] * std
            res.loc[col, :] = top / down

        return res
