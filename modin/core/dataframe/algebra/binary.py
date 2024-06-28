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

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any, Callable, Optional, Union

import numpy as np
import pandas
from pandas.api.types import is_bool_dtype, is_scalar

from modin.error_message import ErrorMessage

from .operator import Operator

if TYPE_CHECKING:
    from pandas._typing import DtypeObj

    from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler


def maybe_compute_dtypes_common_cast(
    first: PandasQueryCompiler,
    second: Union[PandasQueryCompiler, dict, list, tuple, np.ndarray, str, DtypeObj],
    trigger_computations: bool = False,
    axis: int = 0,
    func: Optional[
        Callable[[pandas.DataFrame, pandas.DataFrame], pandas.DataFrame]
    ] = None,
) -> Optional[pandas.Series]:
    """
    Precompute data types for binary operations by finding common type between operands.

    Parameters
    ----------
    first : PandasQueryCompiler
        First operand for which the binary operation would be performed later.
    second : PandasQueryCompiler, dict, list, tuple, np.ndarray, str or DtypeObj
        Second operand for which the binary operation would be performed later.
    trigger_computations : bool, default: False
        Whether to trigger computation of the lazy metadata for `first` and `second`.
        If False is specified this method will return None if any of the operands doesn't
        have materialized dtypes.
    axis : int, default: 0
        Axis to perform the binary operation along.
    func : callable(pandas.DataFrame, pandas.DataFrame) -> pandas.DataFrame, optional
        If specified, will use this function to perform the "try_sample" method
        (see ``Binary.register()`` docs for more details).

    Returns
    -------
    pandas.Series
        The pandas series with precomputed dtypes or None if there's not enough metadata to compute it.

    Notes
    -----
    The dtypes of the operands are supposed to be known.
    """
    if not trigger_computations:
        if not first.frame_has_materialized_dtypes:
            return None

        if isinstance(second, type(first)) and not second.frame_has_materialized_dtypes:
            return None

    dtypes_first = first.dtypes.to_dict()
    if isinstance(second, type(first)):
        dtypes_second = second.dtypes.to_dict()
        columns_first = set(first.columns)
        columns_second = set(second.columns)
        common_columns = columns_first.intersection(columns_second)
        # Here we want to XOR the sets in order to find the columns that do not
        # belong to the intersection, these will be NaN columns in the result
        mismatch_columns = columns_first ^ columns_second
    elif isinstance(second, dict):
        dtypes_second = {
            key: pandas.api.types.pandas_dtype(type(value))
            for key, value in second.items()
        }
        columns_first = set(first.columns)
        columns_second = set(second.keys())
        common_columns = columns_first.intersection(columns_second)
        # Here we want to find the difference between the sets in order to find columns
        # that are missing in the dictionary, this will be NaN columns in the result
        mismatch_columns = columns_first.difference(columns_second)
    else:
        if isinstance(second, (list, tuple)):
            second_dtypes_list = (
                [pandas.api.types.pandas_dtype(type(value)) for value in second]
                if axis == 1
                # Here we've been given a column so it has only one dtype,
                # Infering the dtype using `np.array`, TODO: maybe there's more efficient way?
                else [np.array(second).dtype] * len(dtypes_first)
            )
        elif is_scalar(second) or isinstance(second, np.ndarray):
            try:
                dtype = getattr(second, "dtype", None) or pandas.api.types.pandas_dtype(
                    type(second)
                )
            except TypeError:
                # For example, dtype '<class 'datetime.datetime'>' not understood
                dtype = pandas.Series(second).dtype
            second_dtypes_list = [dtype] * len(dtypes_first)
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
    nan_dtype = pandas.api.types.pandas_dtype(type(np.nan))
    dtypes = None
    if func is not None:
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore")
                df1 = pandas.DataFrame([[1] * len(common_columns)]).astype(
                    {i: dtypes_first[col] for i, col in enumerate(common_columns)}
                )
                df2 = pandas.DataFrame([[1] * len(common_columns)]).astype(
                    {i: dtypes_second[col] for i, col in enumerate(common_columns)}
                )
                dtypes = func(df1, df2).dtypes.set_axis(common_columns)
        # it sometimes doesn't work correctly with strings, so falling back to
        # the "common_cast" method in this case
        except TypeError:
            pass
    if dtypes is None:
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
    dtypes: pandas.Series = pandas.concat(
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
    first: PandasQueryCompiler,
    second: Union[PandasQueryCompiler, Any],
    dtype: DtypeObj,
    trigger_computations: bool = False,
) -> Optional[pandas.Series]:
    """
    Build a ``pandas.Series`` describing dtypes of the result of a binary operation.

    Parameters
    ----------
    first : PandasQueryCompiler
        First operand for which the binary operation would be performed later.
    second : PandasQueryCompiler, list-like or scalar
        Second operand for which the binary operation would be performed later.
    dtype : DtypeObj
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
        if not first.frame_has_columns_cache:
            return None

        if isinstance(second, type(first)) and not second.frame_has_columns_cache:
            return None

    columns_first = set(first.columns)
    if isinstance(second, type(first)):
        columns_second = set(second.columns)
        columns_union = columns_first.union(columns_second)
    else:
        columns_union = columns_first

    dtypes = pandas.Series([dtype] * len(columns_union), index=columns_union)
    return dtypes


def try_compute_new_dtypes(
    first: PandasQueryCompiler,
    second: Union[PandasQueryCompiler, Any],
    infer_dtypes: Optional[str] = None,
    result_dtype: Optional[Union[DtypeObj, str]] = None,
    axis: int = 0,
    func: Optional[
        Callable[[pandas.DataFrame, pandas.DataFrame], pandas.DataFrame]
    ] = None,
) -> Optional[pandas.Series]:
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
    infer_dtypes : {"common_cast", "try_sample", "bool", None}, default: None
        How dtypes should be infered (see ``Binary.register`` doc for more info).
    result_dtype : np.dtype, optional
        NumPy dtype of the result. If not specified it will be inferred from the `infer_dtypes` parameter.
    axis : int, default: 0
        Axis to perform the binary operation along.
    func : callable(pandas.DataFrame, pandas.DataFrame) -> pandas.DataFrame, optional
        A callable to be used for the "try_sample" method.

    Returns
    -------
    pandas.Series or None
    """
    if infer_dtypes is None and result_dtype is None:
        return None

    try:
        if infer_dtypes == "bool" or is_bool_dtype(result_dtype):
            dtypes = maybe_build_dtypes_series(
                first, second, dtype=pandas.api.types.pandas_dtype(bool)
            )
        elif infer_dtypes == "common_cast":
            dtypes = maybe_compute_dtypes_common_cast(
                first, second, axis=axis, func=None
            )
        elif infer_dtypes == "try_sample":
            if func is None:
                raise ValueError(
                    "'func' must be specified if dtypes infering method is 'try_sample'"
                )
            dtypes = maybe_compute_dtypes_common_cast(
                first, second, axis=axis, func=func
            )
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
        func: Callable[..., pandas.DataFrame],
        join_type: str = "outer",
        sort: bool = None,
        labels: str = "replace",
        infer_dtypes: Optional[str] = None,
    ) -> Callable[..., PandasQueryCompiler]:
        """
        Build template binary operator.

        Parameters
        ----------
        func : callable(pandas.DataFrame, [pandas.DataFrame, list-like, scalar]) -> pandas.DataFrame
            Binary function to execute. Have to be able to accept at least two arguments.
        join_type : {'left', 'right', 'outer', 'inner', None}, default: 'outer'
            Type of join that will be used if indices of operands are not aligned.
        sort : bool, default: None
            Whether to sort index and columns or not.
        labels : {"keep", "replace", "drop"}, default: "replace"
            Whether keep labels from left Modin DataFrame, replace them with labels
            from joined DataFrame or drop altogether to make them be computed lazily later.
        infer_dtypes : {"common_cast", "try_sample", "bool", None}, default: None
            How dtypes should be inferred.
                * If "common_cast", casts to common dtype of operand columns.
                * If "try_sample", creates small pandas DataFrames with dtypes of operands and
                  runs the `func` on them to determine output dtypes. If a ``TypeError`` is raised
                  during this process, fallback to "common_cast" method.
                * If "bool", dtypes would be a boolean series with same size as that of operands.
                * If ``None``, do not infer new dtypes (they will be computed manually once accessed).

        Returns
        -------
        callable
            Function that takes query compiler and executes binary operation.
        """

        def caller(
            query_compiler: PandasQueryCompiler,
            other: Union[PandasQueryCompiler, Any],
            broadcast: bool = False,
            *args: tuple,
            dtypes: Optional[Union[DtypeObj, str]] = None,
            **kwargs: dict,
        ) -> PandasQueryCompiler:
            """
            Apply binary `func` to passed operands.

            Parameters
            ----------
            query_compiler : PandasQueryCompiler
                Left operand of `func`.
            other : PandasQueryCompiler, list-like object or scalar
                Right operand of `func`.
            broadcast : bool, default: False
                If `other` is a one-column query compiler, indicates whether it is a Series or not.
                Frames and Series have to be processed differently, however we can't distinguish them
                at the query compiler level, so this parameter is a hint that passed from a high level API.
            *args : tuple,
                Arguments that will be passed to `func`.
            dtypes : "copy", scalar dtype or None, default: None
                Dtypes of the result. "copy" to keep old dtypes and None to compute them on demand.
            **kwargs : dict,
                Arguments that will be passed to `func`.

            Returns
            -------
            PandasQueryCompiler
                Result of binary function.
            """
            axis: int = kwargs.get("axis", 0)
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
                    query_compiler, other, infer_dtypes, dtypes, axis, func
                )

            shape_hint = None
            if isinstance(other, type(query_compiler)):
                if broadcast:
                    if (
                        query_compiler.frame_has_materialized_columns
                        and other.frame_has_materialized_columns
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
                        query_compiler.frame_has_materialized_columns
                        and other.frame_has_materialized_columns
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
                            sort=sort,
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
                        query_compiler.frame_has_materialized_columns
                        and len(query_compiler._modin_frame.columns) == 1
                        and is_scalar(other)
                    ):
                        shape_hint = "column"
                    new_modin_frame = query_compiler._modin_frame.map(
                        func,
                        func_args=(other, *args),
                        func_kwargs=kwargs,
                        dtypes=dtypes,
                        lazy=True,
                    )
                return query_compiler.__constructor__(
                    new_modin_frame, shape_hint=shape_hint
                )

        return caller
