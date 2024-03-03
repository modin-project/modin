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

"""Contains implementations for Merge/Join."""

import pandas
from pandas.core.dtypes.common import is_list_like
from pandas.errors import MergeError

from modin.core.dataframe.base.dataframe.utils import join_columns
from modin.core.dataframe.pandas.metadata import ModinDtypes

from .utils import merge_partitioning


# TODO: add methods for 'join' here
class MergeImpl:
    """Provide implementations for merge/join."""

    @classmethod
    def range_partitioning_merge(cls, left, right, kwargs):
        """
        Execute merge using range-partitioning implementation.

        Parameters
        ----------
        left : PandasQueryCompiler
        right : PandasQueryCompiler
        kwargs : dict
            Keyword arguments for ``pandas.merge()`` function.

        Returns
        -------
        PandasQueryCompiler
        """
        if (
            kwargs.get("left_index", False)
            or kwargs.get("right_index", False)
            or kwargs.get("left_on", None) is not None
            or kwargs.get("left_on", None) is not None
            or kwargs.get("how", "left") not in ("left", "inner")
        ):
            raise NotImplementedError(
                f"The passed parameters are not yet supported by range-partitioning merge: {kwargs=}"
            )

        on = kwargs.get("on", None)
        if on is not None and not isinstance(on, list):
            on = [on]
        if on is None or len(on) > 1:
            raise NotImplementedError(
                f"Merging on multiple columns is not yet supported by range-partitioning merge: {on=}"
            )

        if any(col not in left.columns or col not in right.columns for col in on):
            raise NotImplementedError(
                "Merging on an index level is not yet supported by range-partitioning merge."
            )

        def func(left, right):
            return left.merge(right, **kwargs)

        new_columns, new_dtypes = cls._compute_result_metadata(
            left,
            right,
            on,
            left_on=None,
            right_on=None,
            suffixes=kwargs.get("suffixes", ("_x", "_y")),
        )

        return left.__constructor__(
            left._modin_frame._apply_func_to_range_partitioning_broadcast(
                right._modin_frame,
                func=func,
                key=on,
                new_columns=new_columns,
                new_dtypes=new_dtypes,
            )
            # pandas resets the index of the result unless we were merging on an index level,
            # the current implementation only supports merging on column names, so dropping
            # the index unconditionally
        ).reset_index(drop=True)

    @classmethod
    def row_axis_merge(cls, left, right, kwargs):
        """
        Execute merge using row-axis implementation.

        Parameters
        ----------
        left : PandasQueryCompiler
        right : PandasQueryCompiler
        kwargs : dict
            Keyword arguments for ``pandas.merge()`` function.

        Returns
        -------
        PandasQueryCompiler
        """
        how = kwargs.get("how", "inner")
        on = kwargs.get("on", None)
        left_on = kwargs.get("left_on", None)
        right_on = kwargs.get("right_on", None)
        left_index = kwargs.get("left_index", False)
        right_index = kwargs.get("right_index", False)
        sort = kwargs.get("sort", False)
        right_to_broadcast = right._modin_frame.combine()

        if how in ["left", "inner"] and left_index is False and right_index is False:
            kwargs["sort"] = False

            def should_keep_index(left, right):
                keep_index = False
                if left_on is not None and right_on is not None:
                    keep_index = any(
                        o in left.index.names
                        and o in right_on
                        and o in right.index.names
                        for o in left_on
                    )
                elif on is not None:
                    keep_index = any(
                        o in left.index.names and o in right.index.names for o in on
                    )
                return keep_index

            def map_func(
                left, right, *axis_lengths, kwargs=kwargs, **service_kwargs
            ):  # pragma: no cover
                df = pandas.merge(left, right, **kwargs)

                if kwargs["how"] == "left":
                    partition_idx = service_kwargs["partition_idx"]
                    if len(axis_lengths):
                        if not should_keep_index(left, right):
                            # Doesn't work for "inner" case, since the partition sizes of the
                            # left dataframe may change
                            start = sum(axis_lengths[:partition_idx])
                            stop = sum(axis_lengths[: partition_idx + 1])

                            df.index = pandas.RangeIndex(start, stop)

                return df

            # Want to ensure that these are python lists
            if left_on is not None and right_on is not None:
                left_on = list(left_on) if is_list_like(left_on) else [left_on]
                right_on = list(right_on) if is_list_like(right_on) else [right_on]
            elif on is not None:
                on = list(on) if is_list_like(on) else [on]

            new_columns, new_dtypes = cls._compute_result_metadata(
                left, right, on, left_on, right_on, kwargs.get("suffixes", ("_x", "_y"))
            )

            new_left = left.__constructor__(
                left._modin_frame.broadcast_apply_full_axis(
                    axis=1,
                    func=map_func,
                    enumerate_partitions=how == "left",
                    other=right_to_broadcast,
                    # We're going to explicitly change the shape across the 1-axis,
                    # so we want for partitioning to adapt as well
                    keep_partitioning=False,
                    num_splits=merge_partitioning(
                        left._modin_frame, right._modin_frame, axis=1
                    ),
                    new_columns=new_columns,
                    sync_labels=False,
                    dtypes=new_dtypes,
                    pass_axis_lengths_to_partitions=how == "left",
                )
            )

            # Here we want to understand whether we're joining on a column or on an index level.
            # It's cool if indexes are already materialized so we can easily check that, if not
            # it's fine too, we can also decide that by columns, which tend to be already
            # materialized quite often compared to the indexes.
            keep_index = False
            if left._modin_frame.has_materialized_index:
                keep_index = should_keep_index(left, right)
            else:
                # Have to trigger columns materialization. Hope they're already available at this point.
                if left_on is not None and right_on is not None:
                    keep_index = any(
                        o not in right.columns
                        and o in left_on
                        and o not in left.columns
                        for o in right_on
                    )
                elif on is not None:
                    keep_index = any(
                        o not in right.columns and o not in left.columns for o in on
                    )

            if sort:
                if left_on is not None and right_on is not None:
                    new_left = (
                        new_left.sort_index(axis=0, level=left_on + right_on)
                        if keep_index
                        else new_left.sort_rows_by_column_values(left_on + right_on)
                    )
                elif on is not None:
                    new_left = (
                        new_left.sort_index(axis=0, level=on)
                        if keep_index
                        else new_left.sort_rows_by_column_values(on)
                    )

            return (
                new_left.reset_index(drop=True)
                if not keep_index and (kwargs["how"] != "left" or sort)
                else new_left
            )
        else:
            return left.default_to_pandas(pandas.DataFrame.merge, right, **kwargs)

    @classmethod
    def _compute_result_metadata(cls, left, right, on, left_on, right_on, suffixes):
        """
        Compute columns and dtypes metadata for the result of merge if possible.

        Parameters
        ----------
        left : PandasQueryCompiler
        right : PandasQueryCompiler
        on : label, list of labels or None
            `on` argument that was passed to ``pandas.merge()``.
        left_on : label, list of labels or None
            `left_on` argument that was passed to ``pandas.merge()``.
        right_on : label, list of labels or None
            `right_on` argument that was passed to ``pandas.merge()``.
        suffixes : list of strings
            `suffixes` argument that was passed to ``pandas.merge()``.

        Returns
        -------
        new_columns : pandas.Index or None
            Columns for the result of merge. ``None`` if not enought metadata to compute.
        new_dtypes : ModinDtypes or None
            Dtypes for the result of merge. ``None`` if not enought metadata to compute.
        """
        new_columns = None
        new_dtypes = None

        if not left._modin_frame.has_materialized_columns:
            return new_columns, new_dtypes

        if left_on is None and right_on is None:
            if on is None:
                on = [c for c in left.columns if c in right.columns]
            _left_on, _right_on = on, on
        else:
            if left_on is None or right_on is None:
                raise MergeError(
                    "Must either pass only 'on' or 'left_on' and 'right_on', not combination of them."
                )
            _left_on, _right_on = left_on, right_on

        try:
            new_columns, left_renamer, right_renamer = join_columns(
                left.columns,
                right.columns,
                _left_on,
                _right_on,
                suffixes,
            )
        except NotImplementedError:
            # This happens when one of the keys to join is an index level. Pandas behaviour
            # is really complicated in this case, so we're not computing resulted columns for now.
            pass
        else:
            # renamers may contain columns from 'index', so trying to merge index and column dtypes here
            right_index_dtypes = (
                right.index.dtypes
                if isinstance(right.index, pandas.MultiIndex)
                else pandas.Series([right.index.dtype], index=[right.index.name])
            )
            right_dtypes = pandas.concat([right.dtypes, right_index_dtypes])[
                right_renamer.keys()
            ].rename(right_renamer)

            left_index_dtypes = left._modin_frame._index_cache.maybe_get_dtypes()
            left_dtypes = (
                ModinDtypes.concat([left._modin_frame._dtypes, left_index_dtypes])
                .lazy_get(left_renamer.keys())
                .set_index(list(left_renamer.values()))
            )
            new_dtypes = ModinDtypes.concat([left_dtypes, right_dtypes])

        return new_columns, new_dtypes
