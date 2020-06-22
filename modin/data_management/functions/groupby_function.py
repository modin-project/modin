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
            assert isinstance(
                by, type(query_compiler)
            ), "Can only use groupby reduce with another Query Compiler"
            assert axis == 0, "Can only groupby reduce with axis=0"

            if numeric_only:
                qc = query_compiler.getitem_column_array(
                    query_compiler._modin_frame._numeric_columns(True)
                )
            else:
                qc = query_compiler
            as_index = groupby_args.get("as_index", True)

            def _map(df, other):
                def compute_map(df, other):
                    # Set `as_index` to True to track the metadata of the grouping object
                    # It is used to make sure that between phases we are constructing the
                    # right index and placing columns in the correct order.
                    groupby_args["as_index"] = True
                    other = other.squeeze(axis=axis ^ 1)
                    if isinstance(other, pandas.DataFrame):
                        df = pandas.concat(
                            [df] + [other[[o for o in other if o not in df]]], axis=1
                        )
                        other = list(other.columns)
                    result = map_func(
                        df.groupby(by=other, axis=axis, **groupby_args), **map_args
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
                            new_result = result.drop(columns=col_name)
                            new_result.columns = [
                                col_name if "_modin_groupby_" in c else c
                                for c in new_result.columns
                            ]
                            return new_result
                        else:
                            return result.drop(columns=by_part)
                    return result

                try:
                    return compute_reduce(df)
                # This will happen with Arrow buffer read-only errors. We don't want to copy
                # all the time, so this will try to fast-path the code first.
                except ValueError:
                    return compute_reduce(df.copy())

            if axis == 0:
                new_columns = qc.columns
                new_index = None
            else:
                new_index = query_compiler.index
                new_columns = None
            new_modin_frame = qc._modin_frame.groupby_reduce(
                axis,
                by._modin_frame,
                _map,
                _reduce,
                new_columns=new_columns,
                new_index=new_index,
            )
            return query_compiler.__constructor__(new_modin_frame)

        return caller
