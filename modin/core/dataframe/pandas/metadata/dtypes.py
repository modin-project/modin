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


class ModinDtypes:
    """
    A class that hides the various implementations of the dtypes needed for optimization.

    Parameters
    ----------
    value : sequence or callable
    """

    def __init__(self, value):
        self._value = value

    @property
    def is_materialized(self) -> bool:
        """
        Check if the internal representation is materialized.

        Returns
        -------
        bool
        """
        return isinstance(self._value, pandas.Series)

    def get(self) -> pandas.Series:
        """
        Get the materialized internal representation.

        Returns
        -------
        pandas.Series
        """
        if not self.is_materialized:
            if callable(self._value):
                self._value = self._value()
            else:
                raise NotImplementedError(type(self._value))
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
        return (self.__class__, (self._value,))

    def __getattr__(self, name):
        """
        Redirect access to non-existent attributes to the internal representation.
        This is necessary so that objects of this class in most cases mimic the behavior
        of the ``pandas.Series``. The main limitations of the current approach are type
        checking and the use of this object where pandas dtypes are supposed to be used.

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

    def copy(self) -> "ModinDtypes":
        """
        Copy an object without materializing the internal representation.

        Returns
        -------
        ModinDtypes
        """
        idx_cache = self._value
        if not callable(idx_cache):
            idx_cache = idx_cache.copy()
        return ModinDtypes(idx_cache)

    def __getitem__(self, key):
        if not self.is_materialized:
            self.get()
        return self._value.__getitem__(key)

    def __setitem__(self, key, item):
        if not self.is_materialized:
            self.get()
        self._value.__setitem__(key, item)

    def __iter__(self):
        if not self.is_materialized:
            self.get()
        return iter(self._value)
