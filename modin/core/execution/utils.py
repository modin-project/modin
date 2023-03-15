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

"""Contains utility functions for partitioning."""


def _get_index_and_columns_size(df):  # pragma: no cover
    """
    Get the number of rows and columns of a pandas DataFrame.

    Parameters
    ----------
    df : pandas.DataFrame
        A pandas DataFrame which dimensions are needed.

    Returns
    -------
    int
        The number of rows.
    int
        The number of columns.
    """
    return len(df.index), len(df.columns)


def get_apply_func(get_ip_callable):
    def _apply_func(partition, func, *args, **kwargs):  # pragma: no cover
        """
        Execute a function on the partition in a worker process.

        Parameters
        ----------
        partition : pandas.DataFrame
            A pandas DataFrame the function needs to be executed on.
        func : callable
            The function to perform on the partition.
        *args : list
            Positional arguments to pass to ``func``.
        **kwargs : dict
            Keyword arguments to pass to ``func``.

        Returns
        -------
        pandas.DataFrame
            The resulting pandas DataFrame.
        int
            The number of rows of the resulting pandas DataFrame.
        int
            The number of columns of the resulting pandas DataFrame.
        str
            The node IP address of the worker process.

        Notes
        -----
        Directly passing a call queue entry (i.e. a list of [func, args, kwargs]) instead of
        destructuring it causes a performance penalty.
        """
        try:
            result = func(partition, *args, **kwargs)
        # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
        # don't want the error to propagate to the user, and we want to avoid copying unless
        # we absolutely have to.
        except ValueError:
            result = func(partition.copy(), *args, **kwargs)
        return (
            result,
            len(result) if hasattr(result, "__len__") else 0,
            len(getattr(result, "columns", ())),
            get_ip_callable(),
        )

    return _apply_func


def get_apply_list_of_funcs(get_ip_callable, deserialize_callable):
    def _apply_list_of_funcs(call_queue, partition):  # pragma: no cover
        """
        Execute all operations stored in the call queue on the partition in a worker process.

        Parameters
        ----------
        call_queue : list
            A call queue that needs to be executed on the partition.
        partition : pandas.DataFrame
            A pandas DataFrame the call queue needs to be executed on.

        Returns
        -------
        pandas.DataFrame
            The resulting pandas DataFrame.
        int
            The number of rows of the resulting pandas DataFrame.
        int
            The number of columns of the resulting pandas DataFrame.
        str
            The node IP address of the worker process.
        """
        for func, f_args, f_kwargs in call_queue:
            func = deserialize_callable(func)
            args = deserialize_callable(f_args)
            kwargs = deserialize_callable(f_kwargs)
            try:
                partition = func(partition, *args, **kwargs)
            # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
            # don't want the error to propagate to the user, and we want to avoid copying unless
            # we absolutely have to.
            except ValueError:
                partition = func(partition.copy(), *args, **kwargs)

        return (
            partition,
            len(partition) if hasattr(partition, "__len__") else 0,
            len(partition.columns) if hasattr(partition, "columns") else 0,
            get_ip_callable(),
        )

    return _apply_list_of_funcs
