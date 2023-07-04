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

"""Module houses builder class for Binary operator."""

import numpy as np
import pandas
from pandas.api.types import is_scalar, is_bool_dtype
from typing import Optional

from .operator import Operator
from modin.error_message import ErrorMessage


def coerce_int_to_float64(dtype: np.dtype) -> np.dtype:
    """
    Coerce dtype to float64 if it is a variant of integer.

    If dtype is integer, function returns float64 datatype.
    If not, returns the datatype argument itself.

    Parameters
    ----------
    dtype : np.dtype
        NumPy datatype.

    Returns
    -------
    dtype : np.dtype
        Returns float64 for all int datatypes or returns the datatype itself
        for other types.

    Notes
    -----
    Used to precompute datatype in case of division in pandas.
    """
    if dtype in np.sctypes["int"] + np.sctypes["uint"]:
        return np.dtype(np.float64)
    else:
        return dtype


def maybe_compute_dtypes_common_cast(
    first, second, trigger_computations=False, axis=0
) -> Optional[pandas.Series]:
    """
    Precompute data types for binary operations by finding common type between operands.

    Parameters
    ----------
    first : PandasQueryCompiler
        First operand for which the binary operation would be performed later.
    second : PandasQueryCompiler, list-like or scalar
        Second operand for which the binary operation would be performed later.
    trigger_computations : bool, default: False
        Whether to trigger computation of the lazy metadata for `first` and `second`.
        If False is specified this method will return None if any of the operands doesn't
        have materialized dtypes.
    axis : int, default: 0
        Axis to perform the binary operation along.

    Returns
    -------
    pandas.Series
        The pandas series with precomputed dtypes or None if there's not enough metadata to compute it.

    Notes
    -----
    The dtypes of the operands are supposed to be known.
    """
    if not trigger_computations:
        if not first._modin_frame.has_materialized_dtypes:
            return None

        if (
            isinstance(second, type(first))
            and not second._modin_frame.has_materialized_dtypes
        ):
            return None

    dtypes_first = first._modin_frame.dtypes.to_dict()
    if isinstance(second, type(first)):
        dtypes_second = second._modin_frame.dtypes.to_dict()
        columns_first = set(first.columns)
        columns_second = set(second.columns)
        common_columns = columns_first.intersection(columns_second)
        # Here we want to XOR the sets in order to find the columns that do not
        # belong to the intersection, these will be NaN columns in the result
        mismatch_columns = columns_first ^ columns_second
    elif isinstance(second, dict):
        dtypes_second = {key: type(value) for key, value in second.items()}
        columns_first = set(first.columns)
        columns_second = set(second.keys())
        common_columns = columns_first.intersection(columns_second)
        # Here we want to find the difference between the sets in order to find columns
        # that are missing in the dictionary, this will be NaN columns in the result
        mismatch_columns = columns_first.difference(columns_second)
    else:
        if isinstance(second, (list, tuple)):
            second_dtypes_list = (
                [type(value) for value in second]
                if axis == 1
                # Here we've been given a column so it has only one dtype,
                # Infering the dtype using `np.array`, TODO: maybe there's more efficient way?
                else [np.array(second).dtype] * len(dtypes_first)
            )
        elif is_scalar(second) or isinstance(second, np.ndarray):
            second_dtypes_list = [getattr(second, "dtype", type(second))] * len(
                dtypes_first
            )
        else:
            raise NotImplementedError(
                f"Can't compute common type for {type(first)} and {type(second)}."
            )
        # We verify operands shapes at the front-end, invalid operands shouldn't be
        # propagated to the query compiler level
        ErrorMessage.catch_bugs_and_request_email(
            failure_condition=len(second_dtypes_list) != len(dtypes_first),
            extra_log="Shapes of the operands of a binary operation don't match",
        )
        dtypes_second = {
            key: second_dtypes_list[idx] for idx, key in enumerate(dtypes_first.keys())
        }
        common_columns = first.columns
        mismatch_columns = []

    # If at least one column doesn't match, the result of the non matching column would be nan.
    nan_dtype = np.dtype(type(np.nan))
    dtypes = pandas.Series(
        [
            pandas.core.dtypes.cast.find_common_type(
                [
                    dtypes_first[x],
                    dtypes_second[x],
                ]
            )
            for x in common_columns
        ],
        index=common_columns,
    )
    dtypes = pandas.concat(
        [
            dtypes,
            pandas.Series(
                [nan_dtype] * (len(mismatch_columns)),
                index=mismatch_columns,
            ),
        ]
    )
    return dtypes


def maybe_build_dtypes_series(
    first, second, dtype, trigger_computations=False
) -> Optional[pandas.Series]:
    """
    Build a ``pandas.Series`` describing dtypes of the result of a binary operation.

    Parameters
    ----------
    first : PandasQueryCompiler
        First operand for which the binary operation would be performed later.
    second : PandasQueryCompiler, list-like or scalar
        Second operand for which the binary operation would be performed later.
    dtype : np.dtype
        Dtype of the result.
    trigger_computations : bool, default: False
        Whether to trigger computation of the lazy metadata for `first` and `second`.
        If False is specified this method will return None if any of the operands doesn't
        have materialized columns.

    Returns
    -------
    pandas.Series or None
        The pandas series with precomputed dtypes or None if there's not enough metadata to compute it.

    Notes
    -----
    Finds a union of columns and finds dtypes for all these columns.
    """
    if not trigger_computations:
        if not first._modin_frame.has_columns_cache:
            return None

        if (
            isinstance(second, type(first))
            and not second._modin_frame.has_columns_cache
        ):
            return None

    columns_first = set(first.columns)
    if isinstance(second, type(first)):
        columns_second = set(second.columns)
        columns_union = columns_first.union(columns_second)
    else:
        columns_union = columns_first

    dtypes = pandas.Series([dtype] * len(columns_union), index=columns_union)
    return dtypes


def try_compute_new_dtypes(first, second, infer_dtypes=None, result_dtype=None, axis=0):
    """
    Precompute resulting dtypes of the binary operation if possible.

    The dtypes won't be precomputed if any of the operands doesn't have their dtypes materialized
    or if the second operand type is not supported. Supported types: PandasQueryCompiler, list,
    dict, tuple, np.ndarray.

    Parameters
    ----------
    first : PandasQueryCompiler
        First operand of the binary operation.
    second : PandasQueryCompiler, list-like or scalar
        Second operand of the binary operation.
    infer_dtypes : {"common_cast", "float", "bool", None}, default: None
        How dtypes should be infered (see ``Binary.register`` doc for more info).
    result_dtype : np.dtype, optional
        NumPy dtype of the result. If not specified it will be inferred from the `infer_dtypes` parameter.
    axis : int, default: 0
        Axis to perform the binary operation along.

    Returns
    -------
    pandas.Series or None
    """
    if infer_dtypes is None and result_dtype is None:
        return None

    try:
        if infer_dtypes == "bool" or is_bool_dtype(result_dtype):
            dtypes = maybe_build_dtypes_series(first, second, dtype=np.dtype(bool))
        elif infer_dtypes == "common_cast":
            dtypes = maybe_compute_dtypes_common_cast(first, second, axis=axis)
        elif infer_dtypes == "float":
            dtypes = maybe_compute_dtypes_common_cast(first, second, axis=axis)
            if dtypes is not None:
                dtypes = dtypes.apply(coerce_int_to_float64)
        else:
            # For now we only know how to handle `result_dtype == bool` as that's
            # the only value that is being passed here right now, it's unclear
            # how we should behave in case of an arbitrary dtype, so let's wait
            # for at least one case to appear for this regard.
            dtypes = None
    except NotImplementedError:
        dtypes = None

    return dtypes


class Binary(Operator):
    """Builder class for Binary operator."""

    @classmethod
    def register(
        cls,
        func,
        join_type="outer",
        labels="replace",
        infer_dtypes=None,
    ):
        """
        Build template binary operator.

        Parameters
        ----------
        func : callable(pandas.DataFrame, [pandas.DataFrame, list-like, scalar]) -> pandas.DataFrame
            Binary function to execute. Have to be able to accept at least two arguments.
        join_type : {'left', 'right', 'outer', 'inner', None}, default: 'outer'
            Type of join that will be used if indices of operands are not aligned.
        labels : {"keep", "replace", "drop"}, default: "replace"
            Whether keep labels from left Modin DataFrame, replace them with labels
            from joined DataFrame or drop altogether to make them be computed lazily later.
        infer_dtypes : {"common_cast", "float", "bool", None}, default: None
            How dtypes should be inferred.
                * If "common_cast", casts to common dtype of operand columns.
                * If "float", performs type casting by finding common dtype.
                  If the common dtype is any of the integer types, perform type casting to float.
                  Used in case of truediv.
                * If "bool", dtypes would be a boolean series with same size as that of operands.
                * If ``None``, do not infer new dtypes (they will be computed manually once accessed).

        Returns
        -------
        callable
            Function that takes query compiler and executes binary operation.
        """

        def caller(
            query_compiler, other, broadcast=False, *args, dtypes=None, **kwargs
        ):
            """
            Apply binary `func` to passed operands.

            Parameters
            ----------
            query_compiler : QueryCompiler
                Left operand of `func`.
            other : QueryCompiler, list-like object or scalar
                Right operand of `func`.
            broadcast : bool, default: False
                If `other` is a one-column query compiler, indicates whether it is a Series or not.
                Frames and Series have to be processed differently, however we can't distinguish them
                at the query compiler level, so this parameter is a hint that passed from a high level API.
            *args : args,
                Arguments that will be passed to `func`.
            dtypes : "copy", scalar dtype or None, default: None
                Dtypes of the result. "copy" to keep old dtypes and None to compute them on demand.
            **kwargs : kwargs,
                Arguments that will be passed to `func`.

            Returns
            -------
            QueryCompiler
                Result of binary function.
            """
            axis = kwargs.get("axis", 0)
            if isinstance(other, type(query_compiler)) and broadcast:
                assert (
                    len(other.columns) == 1
                ), "Invalid broadcast argument for `broadcast_apply`, too many columns: {}".format(
                    len(other.columns)
                )
                # Transpose on `axis=1` because we always represent an individual
                # column or row as a single-column Modin DataFrame
                if axis == 1:
                    other = other.transpose()
            if dtypes != "copy":
                dtypes = try_compute_new_dtypes(
                    query_compiler, other, infer_dtypes, dtypes, axis
                )

            shape_hint = None
            if isinstance(other, type(query_compiler)):
                if broadcast:
                    if (
                        query_compiler._modin_frame.has_materialized_columns
                        and other._modin_frame.has_materialized_columns
                    ):
                        if (
                            len(query_compiler.columns) == 1
                            and len(other.columns) == 1
                            and query_compiler.columns.equals(other.columns)
                        ):
                            shape_hint = "column"
                    return query_compiler.__constructor__(
                        query_compiler._modin_frame.broadcast_apply(
                            axis,
                            lambda left, right: func(
                                left, right.squeeze(), *args, **kwargs
                            ),
                            other._modin_frame,
                            join_type=join_type,
                            labels=labels,
                            dtypes=dtypes,
                        ),
                        shape_hint=shape_hint,
                    )
                else:
                    if (
                        query_compiler._modin_frame.has_materialized_columns
                        and other._modin_frame.has_materialized_columns
                    ):
                        if (
                            len(query_compiler.columns) == 1
                            and len(other.columns) == 1
                            and query_compiler.columns.equals(other.columns)
                        ):
                            shape_hint = "column"
                    return query_compiler.__constructor__(
                        query_compiler._modin_frame.n_ary_op(
                            lambda x, y: func(x, y, *args, **kwargs),
                            [other._modin_frame],
                            join_type=join_type,
                            labels=labels,
                            dtypes=dtypes,
                        ),
                        shape_hint=shape_hint,
                    )
            else:
                # TODO: it's possible to chunk the `other` and broadcast them to partitions
                # accordingly, in that way we will be able to use more efficient `._modin_frame.map()`
                if isinstance(other, (dict, list, np.ndarray, pandas.Series)):
                    new_modin_frame = query_compiler._modin_frame.apply_full_axis(
                        axis,
                        lambda df: func(df, other, *args, **kwargs),
                        new_index=query_compiler.index,
                        new_columns=query_compiler.columns,
                        dtypes=dtypes,
                    )
                else:
                    if (
                        query_compiler._modin_frame.has_materialized_columns
                        and len(query_compiler._modin_frame.columns) == 1
                        and is_scalar(other)
                    ):
                        shape_hint = "column"
                    new_modin_frame = query_compiler._modin_frame.map(
                        lambda df: func(df, other, *args, **kwargs),
                        dtypes=dtypes,
                    )
                return query_compiler.__constructor__(
                    new_modin_frame, shape_hint=shape_hint
                )

        return caller
