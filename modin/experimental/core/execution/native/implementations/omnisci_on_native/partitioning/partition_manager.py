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

"""Module provides a partition manager class for ``OmnisciOnNativeDataframe`` frame."""

from modin.pandas.utils import is_scalar
import numpy as np

from modin.core.dataframe.pandas.partitioning.partition_manager import (
    PandasDataframePartitionManager,
)
from ..partitioning.partition import OmnisciOnNativeDataframePartition
from ..omnisci_worker import OmnisciServer
from ..calcite_builder import CalciteBuilder
from ..calcite_serializer import CalciteSerializer
from modin.config import DoUseCalcite

import pyarrow
import pandas
import re


class OmnisciOnNativeDataframePartitionManager(PandasDataframePartitionManager):
    """
    Frame manager for ``OmnisciOnNativeDataframe``.

    This class handles several features of ``OmnisciOnNativeDataframe``:
      - frame always has a single partition
      - frame cannot process some data types
      - frame has to use mangling for index labels
      - frame uses OmniSci storage format for execution
    """

    _partition_class = OmnisciOnNativeDataframePartition

    @classmethod
    def _compute_num_partitions(cls):
        """
        Return a number of partitions a frame should be split to.

        `OmnisciOnNativeDataframe` always has a single partition.

        Returns
        -------
        int
        """
        return 1

    @classmethod
    def from_pandas(cls, df, return_dims=False):
        """
        Create ``OmnisciOnNativeDataframe`` from ``pandas.DataFrame``.

        Parameters
        ----------
        df : pandas.DataFrame
            Source frame.
        return_dims : bool, default: False
            Include resulting dimensions into the returned value.

        Returns
        -------
        tuple
            Tuple holding array of partitions, list of columns with unsupported
            data and optionally partitions' dimensions.
        """

        def tuple_wrapper(obj):
            """
            Wrap non-tuple object into a tuple.

            Parameters
            ----------
            obj : Any.
                Wrapped object.

            Returns
            -------
            tuple
            """
            if not isinstance(obj, tuple):
                obj = (obj,)
            return obj

        if df.empty:
            return (*tuple_wrapper(super().from_pandas(df, return_dims)), [])

        at, unsupported_cols = cls._get_unsupported_cols(df)

        if len(unsupported_cols) > 0:
            # Putting pandas frame into partitions instead of arrow table, because we know
            # that all of operations with this frame will be default to pandas and don't want
            # unnecessaries conversion pandas->arrow->pandas
            return (
                *tuple_wrapper(super().from_pandas(df, return_dims)),
                unsupported_cols,
            )
        else:
            # Since we already have arrow table, putting it into partitions instead
            # of pandas frame, to skip that phase when we will be putting our frame to OmniSci
            return cls.from_arrow(at, return_dims, unsupported_cols)

    @classmethod
    def from_arrow(cls, at, return_dims=False, unsupported_cols=None):
        """
        Build frame from Arrow table.

        Parameters
        ----------
        at : pyarrow.Table
            Input table.
        return_dims : bool, default: False
            True to include dimensions into returned tuple.
        unsupported_cols : list of str, optional
            List of columns holding unsupported data. If None then
            check all columns to compute the list.

        Returns
        -------
        tuple
            Tuple holding array of partitions, list of columns with unsupported
            data and optionally partitions' dimensions.
        """
        put_func = cls._partition_class.put_arrow

        parts = [[put_func(at)]]
        if unsupported_cols is None:
            _, unsupported_cols = cls._get_unsupported_cols(at)

        if not return_dims:
            return np.array(parts), unsupported_cols
        else:
            row_lengths = [at.num_rows]
            col_widths = [at.num_columns]
            return np.array(parts), row_lengths, col_widths, unsupported_cols

    @classmethod
    def _get_unsupported_cols(cls, obj):
        """
        Return a list of columns with unsupported by OmniSci data types.

        Parameters
        ----------
        obj : pandas.DataFrame or pyarrow.Table
            Object to inspect on unsupported column types.

        Returns
        -------
        tuple
            Arrow representation of `obj` (for future using) and a list of
            unsupported columns.
        """
        if isinstance(obj, (pandas.Series, pandas.DataFrame)):
            # picking first rows from cols with `dtype="object"` to check its actual type,
            # in case of homogen columns that saves us unnecessary convertion to arrow table
            cols = [name for name, col in obj.dtypes.items() if col == "object"]
            type_samples = obj.iloc[0][cols]

            unsupported_cols = [
                name
                for name, col in type_samples.items()
                if not isinstance(col, str)
                and not (is_scalar(col) and pandas.isna(col))
            ]

            if len(unsupported_cols) > 0:
                return None, unsupported_cols

            try:
                at = pyarrow.Table.from_pandas(obj)
            except (pyarrow.lib.ArrowTypeError, pyarrow.lib.ArrowInvalid) as e:
                regex = r"Conversion failed for column ([^\W]*)"
                unsupported_cols = []
                for msg in e.args:
                    match = re.findall(regex, msg)
                    unsupported_cols.extend(match)

                if len(unsupported_cols) == 0:
                    unsupported_cols = obj.columns
                return None, unsupported_cols
            else:
                obj = at

        def is_supported_dtype(dtype):
            """Check whether the passed pyarrow `dtype` is supported by OmniSci."""
            if (
                pyarrow.types.is_string(dtype)
                or pyarrow.types.is_time(dtype)
                or pyarrow.types.is_dictionary(dtype)
            ):
                return True
            try:
                pandas_dtype = dtype.to_pandas_dtype()
                return pandas_dtype != np.dtype("O")
            except NotImplementedError:
                return False

        return (
            obj,
            [field.name for field in obj.schema if not is_supported_dtype(field.type)],
        )

    @classmethod
    def run_exec_plan(cls, plan, index_cols, dtypes, columns):
        """
        Run execution plan in OmniSci storage format to materialize frame.

        Parameters
        ----------
        plan : DFAlgNode
            A root of an execution plan tree.
        index_cols : list of str
            A list of index columns.
        dtypes : pandas.Index
            Column data types.
        columns : list of str
            A frame column names.

        Returns
        -------
        np.array
            Created frame's partitions.
        """
        omniSession = OmnisciServer()

        # First step is to make sure all partitions are in OmniSci.
        frames = plan.collect_frames()
        for frame in frames:
            if frame._partitions.size != 1:
                raise NotImplementedError(
                    "OmnisciOnNative engine doesn't suport partitioned frames"
                )
            for p in frame._partitions.flatten():
                if p.frame_id is None:
                    obj = p.get()
                    if isinstance(obj, (pandas.DataFrame, pandas.Series)):
                        p.frame_id = omniSession.put_pandas_to_omnisci(obj)
                    else:
                        assert isinstance(obj, pyarrow.Table)
                        p.frame_id = omniSession.put_arrow_to_omnisci(obj)

        calcite_plan = CalciteBuilder().build(plan)
        calcite_json = CalciteSerializer().serialize(calcite_plan)

        cmd_prefix = "execute relalg "

        if DoUseCalcite.get():
            cmd_prefix = "execute calcite "

        curs = omniSession.executeRA(cmd_prefix + calcite_json)
        assert curs
        if hasattr(curs, "getArrowTable"):
            at = curs.getArrowTable()
        else:
            rb = curs.getArrowRecordBatch()
            assert rb is not None
            at = pyarrow.Table.from_batches([rb])
        assert at is not None

        res = np.empty((1, 1), dtype=np.dtype(object))
        # workaround for https://github.com/modin-project/modin/issues/1851
        if DoUseCalcite.get():
            at = at.rename_columns(["F_" + str(c) for c in columns])
        res[0][0] = cls._partition_class.put_arrow(at)

        return res

    @classmethod
    def _names_from_index_cols(cls, cols):
        """
        Get index labels.

        Deprecated.

        Parameters
        ----------
        cols : list of str
            Index columns.

        Returns
        -------
        list of str
        """
        if len(cols) == 1:
            return cls._name_from_index_col(cols[0])
        return [cls._name_from_index_col(n) for n in cols]

    @classmethod
    def _name_from_index_col(cls, col):
        """
        Get index label.

        Deprecated.

        Parameters
        ----------
        col : str
            Index column.

        Returns
        -------
        str
        """
        if col.startswith("__index__"):
            return None
        return col

    @classmethod
    def _maybe_scalar(cls, lst):
        """
        Transform list with a single element to scalar.

        Deprecated.

        Parameters
        ----------
        lst : list
            Input list.

        Returns
        -------
        Any
        """
        if len(lst) == 1:
            return lst[0]
        return lst
