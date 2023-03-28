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

"""Module contains class ModinIndex."""

import pandas
from pandas.core.indexes.api import ensure_index


class ModinIndex:
    """
    A class that hides the various implementations of the index needed for optimization.

    Parameters
    ----------
    value : sequence or callable
    """

    def __init__(self, value):
        if callable(value):
            self._value = value
        else:
            self._value = ensure_index(value)
        self._lengths_cache = None

    @property
    def is_materialized(self) -> bool:
        """
        Check if the internal representation is materialized.

        Returns
        -------
        bool
        """
        return isinstance(self._value, pandas.Index)

    def get(self, return_lengths=False) -> pandas.Index:
        """
        Get the materialized internal representation.

        Parameters
        ----------
        return_lengths : bool, default: False
            In some cases, during the index calculation, it's possible to get
            the lengths of the partitions. This flag allows this data to be used
            for optimization.

        Returns
        -------
        pandas.Index
        """
        if not self.is_materialized:
            if callable(self._value):
                index, self._lengths_cache = self._value()
                self._value = ensure_index(index)
            else:
                raise NotImplementedError(type(self._value))
        if return_lengths:
            return self._value, self._lengths_cache
        else:
            return self._value

    def __len__(self):
        """
        Redirect the 'len' request to the internal representation.

        Returns
        -------
        int

        Notes
        -----
        Executing this function materializes the data.
        """
        if not self.is_materialized:
            self.get()
        return len(self._value)

    def __reduce__(self):
        """
        Serialize an object of this class.

        Returns
        -------
        tuple

        Notes
        -----
        The default implementation generates a recursion error. In a short:
        during the construction of the object, `__getattr__` function is called, which
        is not intended to be used in situations where the object is not initialized.
        """
        if self._lengths_cache is not None:
            return (self.__class__, (lambda: (self._value, self._lengths_vache),))
        return (self.__class__, (self._value,))

    def __getattr__(self, name):
        """
        Redirect access to non-existent attributes to the internal representation.

        This is necessary so that objects of this class in most cases mimic the behavior
        of the ``pandas.Index``. The main limitations of the current approach are type
        checking and the use of this object where pandas indexes are supposed to be used.

        Parameters
        ----------
        name : str
            Attribute name.

        Returns
        -------
        object
            Attribute.

        Notes
        -----
        Executing this function materializes the data.
        """
        if not self.is_materialized:
            self.get()
        return self._value.__getattribute__(name)

    def copy(self) -> "ModinIndex":
        """
        Copy an object without materializing the internal representation.

        Returns
        -------
        ModinIndex
        """
        idx_cache = self._value
        if not callable(idx_cache):
            idx_cache = idx_cache.copy()
        return ModinIndex(idx_cache)
