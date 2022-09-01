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

"""Module for interface for class compatibility layer for DataFrame."""

from .base import BaseCompatibleBasePandasDataset


class BaseCompatibleDataFrame(BaseCompatibleBasePandasDataset):
    """Interface for compatibility layer for DataFrame."""

    def applymap(self, *args, **kwargs):  # noqa: GL08
        pass

    def apply(self, *args, **kwargs):  # noqa: GL08
        pass

    def info(self, *args, **kwargs):  # noqa: GL08
        pass

    def pivot_table(self, *args, **kwargs):  # noqa: GL08
        pass

    def prod(self, *args, **kwargs):  # noqa: GL08
        pass

    def replace(self, *args, **kwargs):  # noqa: GL08
        pass

    def sum(self, *args, **kwargs):  # noqa: GL08
        pass

    def to_parquet(self, *args, **kwargs):  # noqa: GL08
        pass

    def _disambiguate_axes_labels(self, axis, index, columns, labels):
        """
        Disambiguate axes labels from `index`, `columns` and `labels`.

        Parameters
        ----------
        axis : int, str or pandas._libs.lib.NoDefault
            Axis along which `labels` are specified if they're not None.
        index : pandas.Index or array-like or None
            Optional row labels.
        columns : pandas.Index or array-like or None
            Optional column labels.
        labels : pandas.Index or array-like or None
            Labels along `axis`.

        Returns
        -------
        array-like, array-like
            Pair of row and column labels.
        """
        if labels is not None:
            if self._get_axis_number(axis) == 0:
                index = labels
            else:
                columns = labels
        return index, columns
