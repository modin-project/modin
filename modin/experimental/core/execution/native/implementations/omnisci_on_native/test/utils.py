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

import pytest
import modin.pandas as pd
from modin.utils import try_cast_to_pandas
import pandas
import datetime
import numpy as np
from pandas.api.types import is_datetime64_any_dtype
import pyarrow as pa

from modin.pandas.test.utils import (
    df_equals,
    io_ops_bad_exc,
    eval_io as general_eval_io,
)
from ..df_algebra import FrameNode

from modin.experimental.core.execution.native.implementations.omnisci_on_native.omnisci_worker import (
    OmnisciServer,
)


def eval_io(
    fn_name,
    comparator=df_equals,
    cast_to_str=False,
    check_exception_type=True,
    raising_exceptions=io_ops_bad_exc,
    check_kwargs_callable=True,
    modin_warning=None,
    md_extra_kwargs=None,
    *args,
    **kwargs,
):
    """
    Evaluate I/O operation and do equality check after importing Modin's data to OmniSci.

    Notes
    -----
    For parameters description please refer to ``modin.pandas.test.utils.eval_io``.
    """

    def omnisci_comparator(df1, df2):
        """Evaluate equality comparison of the passed frames after importing the Modin's one to OmniSci."""
        with ForceOmnisciImport(df1, df2):
            # Aligning DateTime dtypes because of the bug related to the `parse_dates` parameter:
            # https://github.com/modin-project/modin/issues/3485
            df1, df2 = align_datetime_dtypes(df1, df2)
            comparator(df1, df2)

    general_eval_io(
        fn_name,
        comparator=omnisci_comparator,
        cast_to_str=cast_to_str,
        check_exception_type=check_exception_type,
        raising_exceptions=raising_exceptions,
        check_kwargs_callable=check_kwargs_callable,
        modin_warning=modin_warning,
        md_extra_kwargs=md_extra_kwargs,
        *args,
        **kwargs,
    )


def align_datetime_dtypes(*dfs):
    """
    Make all of the passed frames have DateTime dtype for the same columns.

    Cast column type of the certain frame to the DateTime type if any frame in
    the `dfs` sequence has DateTime type for this column.

    Parameters
    ----------
    *dfs : iterable of DataFrames
        DataFrames to align DateTime dtypes.

    Notes
    -----
    Passed Modin frames may be casted to pandas in the result.
    """
    datetime_cols = {}
    time_cols = set()
    for df in dfs:
        for col, dtype in df.dtypes.items():
            # If we already decided to cast this column to DateTime no more actions are needed
            if col not in datetime_cols and is_datetime64_any_dtype(dtype):
                datetime_cols[col] = dtype
            # datetime.time is considered to be an 'object' dtype in pandas that's why
            # we have to explicitly check the values type in the column
            elif (
                dtype == np.dtype("O")
                and col not in time_cols
                # OmniSci has difficulties with empty frames, so explicitly skip them
                # https://github.com/modin-project/modin/issues/3428
                and len(df) > 0
                and all(
                    isinstance(val, datetime.time) or pandas.isna(val)
                    for val in df[col]
                )
            ):
                time_cols.add(col)

    if len(datetime_cols) == 0 and len(time_cols) == 0:
        return dfs

    def convert_to_time(value):
        """Convert passed value to `datetime.time`."""
        if isinstance(value, datetime.time):
            return value
        elif isinstance(value, str):
            return datetime.time.fromisoformat(value)
        else:
            return datetime.time(value)

    time_cols_list = list(time_cols)
    casted_dfs = []
    for df in dfs:
        # OmniSci has difficulties with casting to certain dtypes (i.e. datetime64),
        # so casting it to pandas
        pandas_df = try_cast_to_pandas(df)
        if datetime_cols:
            pandas_df = pandas_df.astype(datetime_cols)
        if time_cols:
            pandas_df[time_cols_list] = pandas_df[time_cols_list].applymap(
                convert_to_time
            )
        casted_dfs.append(pandas_df)

    return casted_dfs


class ForceOmnisciImport:
    """
    Trigger import execution for Modin DataFrames obtained by OmniSci engine if already not.

    When using as a context class also cleans up imported tables at the end of the context.

    Parameters
    ----------
    *dfs : iterable
        DataFrames to trigger import.
    """

    def __init__(self, *dfs):
        self._imported_frames = []
        for df in dfs:
            if not isinstance(df, (pd.DataFrame, pd.Series)):
                continue
            df.shape  # to trigger real execution
            if df.empty:
                continue
            partition = df._query_compiler._modin_frame._partitions[0][0]
            if partition.frame_id is not None:
                continue
            frame = partition.get()
            if isinstance(frame, (pandas.DataFrame, pandas.Series)):
                frame_id = OmnisciServer().put_pandas_to_omnisci(frame)
            elif isinstance(frame, pa.Table):
                frame_id = OmnisciServer().put_arrow_to_omnisci(frame)
            else:
                raise TypeError(
                    f"Unexpected storage format, expected pandas.DataFrame or pyarrow.Table, got: {type(frame)}."
                )
            partition.frame_id = frame_id
            self._imported_frames.append((df, frame_id))

    def __enter__(self):
        return self

    def export_frames(self):
        """
        Export tables from OmniSci that was imported by this instance.

        Returns
        -------
        list
            A list of Modin DataFrames whose payload is ``pyarrow.Table``
            that was just exported from OmniSci.
        """
        result = []
        for df, frame_id in self._imported_frames:
            # Append `TransformNode`` selecting all the columns (SELECT * FROM frame_id)
            df = df[df.columns.tolist()]
            modin_frame = df._query_compiler._modin_frame
            # Forcibly executing plan via OmniSci. We can't use `modin_frame._execute()` here
            # as it has a chance of running via pyarrow bypassing OmniSci
            new_partitions = modin_frame._partition_mgr_cls.run_exec_plan(
                modin_frame._op,
                modin_frame._index_cols,
                modin_frame._dtypes,
                modin_frame._table_cols,
            )
            modin_frame._partitions = new_partitions
            modin_frame._op = FrameNode(modin_frame)
            result.append(df)
        return result

    def __exit__(self, exc_type, exc_val, exc_tb):
        for df, frame_id in self._imported_frames:
            actual_frame_id = df._query_compiler._modin_frame._partitions[0][0].frame_id
            OmnisciServer().executeDDL(f"DROP TABLE IF EXISTS {frame_id}")
            if actual_frame_id == frame_id:
                df._query_compiler._modin_frame._partitions[0][0].frame_id = None
        self._imported_frames = []


def set_execution_mode(frame, mode, recursive=False):
    """
    Enable execution mode assertions for the passed frame.

    Enabled execution mode checks mean, that the frame raises an AssertionError
    if the execution flow is out of the scope of the selected mode.

    Parameters
    ----------
    frame : DataFrame or Series
        Modin frame to set execution mode at.
    mode : {None, "lazy", "arrow"}
        Execution mode to set:
            - "lazy": only delayed computations.
            - "arrow": only computations via Pyarrow.
            - None: allow any type of computations.
    recursive : bool, default: False
        Whether to set the specified execution mode for every frame
        in the delayed computation tree.
    """
    if isinstance(frame, (pd.Series, pd.DataFrame)):
        frame = frame._query_compiler._modin_frame
    frame._force_execution_mode = mode
    if recursive and hasattr(frame._op, "input"):
        for child in frame._op.input:
            set_execution_mode(child, mode, True)


def run_and_compare(
    fn,
    data,
    data2=None,
    force_lazy=True,
    force_arrow_execute=False,
    allow_subqueries=False,
    comparator=df_equals,
    **kwargs,
):
    """Verify equality of the results of the passed function executed against pandas and modin frame."""

    def run_modin(
        fn,
        data,
        data2,
        force_lazy,
        force_arrow_execute,
        allow_subqueries,
        constructor_kwargs,
        **kwargs,
    ):
        kwargs["df1"] = pd.DataFrame(data, **constructor_kwargs)
        kwargs["df2"] = pd.DataFrame(data2, **constructor_kwargs)
        kwargs["df"] = kwargs["df1"]

        if force_lazy:
            set_execution_mode(kwargs["df1"], "lazy")
            set_execution_mode(kwargs["df2"], "lazy")
        elif force_arrow_execute:
            set_execution_mode(kwargs["df1"], "arrow")
            set_execution_mode(kwargs["df2"], "arrow")

        exp_res = fn(lib=pd, **kwargs)

        if force_arrow_execute:
            set_execution_mode(exp_res, "arrow", allow_subqueries)
        elif force_lazy:
            set_execution_mode(exp_res, None, allow_subqueries)

        return exp_res

    constructor_kwargs = kwargs.pop("constructor_kwargs", {})
    try:
        kwargs["df1"] = pandas.DataFrame(data, **constructor_kwargs)
        kwargs["df2"] = pandas.DataFrame(data2, **constructor_kwargs)
        kwargs["df"] = kwargs["df1"]
        ref_res = fn(lib=pandas, **kwargs)
    except Exception as e:
        with pytest.raises(type(e)):
            exp_res = run_modin(
                fn=fn,
                data=data,
                data2=data2,
                force_lazy=force_lazy,
                force_arrow_execute=force_arrow_execute,
                allow_subqueries=allow_subqueries,
                constructor_kwargs=constructor_kwargs,
                **kwargs,
            )
            _ = exp_res.index
    else:
        exp_res = run_modin(
            fn=fn,
            data=data,
            data2=data2,
            force_lazy=force_lazy,
            force_arrow_execute=force_arrow_execute,
            allow_subqueries=allow_subqueries,
            constructor_kwargs=constructor_kwargs,
            **kwargs,
        )

        # Currently, strings are converted to categories when exported from OmniSci,
        # this makes the equality comparison fail. Converting string cols back to
        # their original dtypes until the issue is resolved:
        # https://github.com/modin-project/modin/issues/2747
        if isinstance(exp_res, pd.DataFrame):
            external_dtypes = exp_res.dtypes
            exp_res = try_cast_to_pandas(exp_res)
            internal_dtypes = exp_res.dtypes

            new_schema = {}
            for col in exp_res.columns:
                if (
                    internal_dtypes[col] == "category"
                    and external_dtypes[col] != "category"
                ):
                    new_schema[col] = external_dtypes[col]
            exp_res = exp_res.astype(new_schema)

        comparator(ref_res, exp_res)
