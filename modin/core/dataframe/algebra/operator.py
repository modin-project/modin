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

"""Module contains an interface for operator builder classes."""

from typing import Optional


class Operator(object):
    """Interface for building operators that can execute in parallel across partitions."""

    def __init__(self):
        raise ValueError(
            "Please use {}.register instead of the constructor".format(
                type(self).__name__
            )
        )

    @classmethod
    def register(cls, func, **kwargs):
        """
        Build operator that applies source function across the entire dataset.

        Parameters
        ----------
        func : callable
            Source function.
        **kwargs : kwargs
            Kwargs that will be passed to the builder function.

        Returns
        -------
        callable
        """
        raise NotImplementedError("Please implement in child class")

    @classmethod
    def validate_axis(cls, axis: Optional[int]) -> int:
        """
        Ensure that axis to apply function on has valid value.

        Parameters
        ----------
        axis : int, optional
            0 or None means apply on index, 1 means apply on columns.

        Returns
        -------
        int
            Integer representation of given axis.
        """
        return 0 if axis is None else axis

    @classmethod
    def apply(
        cls, df, func, func_args=None, func_kwargs=None, _return_type=None, **kwargs
    ):
        operator = cls.register(func, **kwargs)

        func_args = tuple() if func_args is None else func_args
        func_kwargs = dict() if func_kwargs is None else func_kwargs

        qc_result = operator(df._query_compiler, *func_args, **func_kwargs)

        if _return_type is None:
            _return_type = type(df)

        return _return_type(query_compiler=qc_result)


def apply_operator(
    df,
    operator_cls,
    func,
    return_type=None,
    func_args=None,
    func_kwargs=None,
    *args,
    **kwargs
):
    """
    Apply a function to a modin DataFrame using the passed operator.

    Parameters
    ----------
    df : modin.pandas.DataFrame or modin.pandas.Series
        DataFrame object to apply the operator against.
    operator_cls : Operator
        Operator describing how to apply the function.
    func : callable(pandas.DataFrame, *args, **kwargs) -> pandas.DataFrame
        A function to apply.
    return_type : type, optional
        A class that takes the ``query_compiler`` keyword argument. If not specified
        will be identical to the type of the passed `df`.
    func_args : tuple, optional
    func_kwargs : dict, optional

    Returns
    -------
    return_type
    """
    operator = operator_cls.register(func, *args, **kwargs)

    func_args = tuple() if func_args is None else func_args
    func_kwargs = dict() if func_kwargs is None else func_kwargs

    qc_result = operator(df._query_compiler, *func_args, **func_kwargs)

    if return_type is None:
        return_type = type(df)

    return return_type.__constructor__(query_compiler=qc_result)
