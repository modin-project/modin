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

import pandas

from .mapreducefunction import MapReduceFunction
from modin.utils import try_cast_to_pandas


class GroupbyReduceFunction(MapReduceFunction):
    @classmethod
    def call(cls, map_func, reduce_func, *call_args, **call_kwds):
        def caller(
            query_compiler,
            by,
            axis,
            groupby_args,
            map_args,
            reduce_args=None,
            numeric_only=True,
            drop=False,
        ):
            # breakpoint()
            if not isinstance(by, (type(query_compiler), str)):
                by = try_cast_to_pandas(by)
                if isinstance(by, list):
                    by = [
                        (
                            o.squeeze()
                            if o.columns[0] not in query_compiler.columns
                            else o.columns[0]
                        )
                        if isinstance(o, pandas.DataFrame)
                        else o
                        for o in by
                    ]
                return query_compiler.default_to_pandas(
                    lambda df: map_func(
                        df.groupby(by=by, axis=axis, **groupby_args), **map_args
                    )
                )
            assert axis == 0, "Can only groupby reduce with axis=0"

            if numeric_only:
                qc = query_compiler.getitem_column_array(
                    query_compiler._modin_frame._numeric_columns(True)
                )
            else:
                qc = query_compiler
            # since we're going to modify `groupby_args` dict in a `compute_map`,
            # we want to copy it to not propagate these changes into source dict, in case
            # of unsuccessful end of function
            groupby_args = groupby_args.copy()

            as_index = groupby_args.get("as_index", True)
            observed = groupby_args.get("observed", False)

            if isinstance(by, str):

                def _map(df):
                    # Set `as_index` to True to track the metadata of the grouping
                    # object It is used to make sure that between phases we are
                    # constructing the right index and placing columns in the correct
                    # order.
                    groupby_args["as_index"] = True
                    groupby_args["observed"] = True

                    result = map_func(
                        df.groupby(by=by, axis=axis, **groupby_args), **map_args
                    )
                    # The _modin_groupby_ prefix indicates that this is the first
                    # partition, and since we may need to insert the grouping data in
                    # the reduce phase
                    if (
                        not isinstance(result.index, pandas.MultiIndex)
                        and result.index.name is not None
                        and result.index.name in result.columns
                    ):
                        result.index.name = "{}{}".format(
                            "_modin_groupby_", result.index.name
                        )
                    return result

            else:

                def _map(df, other):
                    def compute_map(df, other):
                        # Set `as_index` to True to track the metadata of the grouping object
                        # It is used to make sure that between phases we are constructing the
                        # right index and placing columns in the correct order.
                        groupby_args["as_index"] = True
                        groupby_args["observed"] = True

                        other = other.squeeze(axis=axis ^ 1)
                        if isinstance(other, pandas.DataFrame):
                            df = pandas.concat(
                                [df] + [other[[o for o in other if o not in df]]],
                                axis=1,
                            )
                            other = list(other.columns)
                        result = map_func(
                            df.groupby(by=other, axis=axis, **groupby_args), **map_args
                        )
                        # if `other` has category dtype, then pandas will drop that
                        # column after groupby, inserting it back to correctly process
                        # reduce phase
                        if (
                            drop
                            and not as_index
                            and isinstance(other, pandas.Series)
                            and isinstance(other.dtype, pandas.CategoricalDtype)
                            and result.index.name is not None
                            and result.index.name not in result.columns
                        ):
                            result.insert(
                                loc=0, column=result.index.name, value=result.index
                            )
                        # The _modin_groupby_ prefix indicates that this is the first partition,
                        # and since we may need to insert the grouping data in the reduce phase
                        if (
                            not isinstance(result.index, pandas.MultiIndex)
                            and result.index.name is not None
                            and result.index.name in result.columns
                        ):
                            result.index.name = "{}{}".format(
                                "_modin_groupby_", result.index.name
                            )
                        return result

                    try:
                        return compute_map(df, other)
                    # This will happen with Arrow buffer read-only errors. We don't want to copy
                    # all the time, so this will try to fast-path the code first.
                    except ValueError:
                        return compute_map(df.copy(), other.copy())

            def _reduce(df):
                def compute_reduce(df):
                    other_len = len(df.index.names)
                    df = df.reset_index(drop=False)
                    # See note above about setting `as_index`
                    groupby_args["as_index"] = as_index
                    groupby_args["observed"] = observed
                    if other_len > 1:
                        by_part = list(df.columns[0:other_len])
                    else:
                        by_part = df.columns[0]
                    result = reduce_func(
                        df.groupby(by=by_part, axis=axis, **groupby_args), **reduce_args
                    )
                    if (
                        not isinstance(result.index, pandas.MultiIndex)
                        and result.index.name is not None
                        and "_modin_groupby_" in result.index.name
                    ):
                        result.index.name = result.index.name[len("_modin_groupby_") :]
                    if isinstance(by_part, str) and by_part in result.columns:
                        if "_modin_groupby_" in by_part and drop:
                            col_name = by_part[len("_modin_groupby_") :]
                            new_result = result.drop(columns=col_name, errors="ignore")
                            new_result.columns = [
                                col_name if "_modin_groupby_" in c else c
                                for c in new_result.columns
                            ]
                            return new_result
                        else:
                            return (
                                result.drop(columns=by_part)
                                if call_kwds.get("method", None) != "size"
                                else result
                            )
                    return result

                try:
                    return compute_reduce(df)
                # This will happen with Arrow buffer read-only errors. We don't want to copy
                # all the time, so this will try to fast-path the code first.
                except ValueError:
                    return compute_reduce(df.copy())

            # TODO: try to precompute `new_index` and `new_columns`
            if isinstance(by, str):
                new_modin_frame = qc._modin_frame._map_reduce(
                    axis, _map, reduce_func=_reduce, preserve_index=False
                )
            else:
                new_modin_frame = qc._modin_frame.groupby_reduce(
                    axis, by._modin_frame, _map, _reduce
                )
            result = query_compiler.__constructor__(new_modin_frame)
            if result.index.name == "__reduced__":
                result.index.name = None
            return result

        return caller
