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

"""Implement pandas plotting API."""

from pandas import plotting as pdplot

from modin.logging import ClassLogger
from modin.pandas.io import to_pandas
from modin.utils import instancer

from .dataframe import DataFrame


@instancer
class Plotting(ClassLogger):
    """Wrapper of pandas plotting module."""

    def __dir__(self):
        """
        Enable tab completion of plotting library.

        Returns
        -------
        list
            List of attributes in `self`.
        """
        return dir(pdplot)

    def __getattribute__(self, item):
        """
        Convert any Modin DataFrames in parameters to pandas so that they can be plotted normally.

        Parameters
        ----------
        item : str
            Attribute to look for.

        Returns
        -------
        object
            If attribute is found in pandas.plotting, and it is a callable, a wrapper function is
            returned which converts its arguments to pandas and calls a function pandas.plotting.`item`
            on these arguments.
            If attribute is found in pandas.plotting but it is not a callable, returns it.
            Otherwise function tries to look for an attribute in `self`.
        """
        if hasattr(pdplot, item):
            func = getattr(pdplot, item)
            if callable(func):

                def wrap_func(*args, **kwargs):
                    """Convert Modin DataFrames to pandas then call the function."""
                    args = tuple(
                        arg if not isinstance(arg, DataFrame) else to_pandas(arg)
                        for arg in args
                    )
                    kwargs = {
                        kwd: val if not isinstance(val, DataFrame) else to_pandas(val)
                        for kwd, val in kwargs.items()
                    }
                    return func(*args, **kwargs)

                return wrap_func
            else:
                return func
        else:
            return object.__getattribute__(self, item)
