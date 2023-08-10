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

import uuid
import pandas
from pandas.core.indexes.api import ensure_index


class ModinIndex:
    """
    A class that hides the various implementations of the index needed for optimization.

    Parameters
    ----------
    value : sequence or callable
    value_ref : optional
        If `value` is a callable that takes an actual object to retrieve index labels from,
        then you can specify this lambda argument via this parameter.
    """

    def __init__(self, value, value_ref=None):
        if callable(value):
            self._value = value
        else:
            self._value = ensure_index(value)
        self._value_ref = value_ref
        self._lengths_cache = None
        # index/lengths ID's for faster comparison between other ModinIndex objects,
        # these should be propagated to the copies of the index
        self._index_id = uuid.uuid4()
        self._lengths_id = uuid.uuid4()

    @property
    def has_value_ref(self) -> bool:
        """
        Check whether the stored lambda requires an object to be passed.

        Returns
        -------
        bool
        """
        return self._value_ref is not None

    def update_ref(self, new_ref):
        """
        Update/set an argument to be passed to the lambda computing actual indices.

        Parameters
        ----------
        new_ref : object
        """
        self._value_ref = new_ref

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
                index, self._lengths_cache = (
                    self._value(self._value_ref)
                    if self.has_value_ref
                    else self._value()
                )
                self._value = ensure_index(index)
                # release the reference to be garbage collected
                self._value_ref = None
            else:
                raise NotImplementedError(type(self._value))
        if return_lengths:
            return self._value, self._lengths_cache
        else:
            return self._value

    def equals(self, other: "ModinIndex") -> bool:
        """
        Check equality of the index values.

        Parameters
        ----------
        other : ModinIndex

        Returns
        -------
        bool
            The result of the comparison.
        """
        if self._id == other._id:
            return True

        if not self.is_materialized:
            self.get()

        if not other.is_materialized:
            other.get()

        return self._value.equals(other._value)

    def compare_partition_lengths_if_possible(self, other: "ModinIndex"):
        """
        Compare the partition lengths cache for the index being stored if possible.

        The ``ModinIndex`` object may sometimes store the information about partition
        lengths along the axis the index belongs to. If both `self` and `other` have
        this information or it can be inferred from them, the method returns
        a boolean - the result of the comparison, otherwise it returns ``None``
        as an indication that the comparison cannot be made.

        Parameters
        ----------
        other : ModinIndex

        Returns
        -------
        bool or None
            The result of the comparison if both `self` and `other` contain
            the lengths data, ``None`` otherwise.
        """
        if self._lengths_id == other._lengths_id:
            return True

        can_extract_lengths_from_self = self._lengths_cache is not None or callable(
            self._value
        )
        can_extract_lengths_from_other = other._lengths_cache is not None or callable(
            other._value
        )
        if can_extract_lengths_from_self and can_extract_lengths_from_other:
            return self.get(return_lengths=True)[1] == other.get(return_lengths=True)[1]
        return None

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

    def copy(self, copy_lengths=False) -> "ModinIndex":
        """
        Copy an object without materializing the internal representation.

        Parameters
        ----------
        copy_lengths : bool, default: True
            Whether to copy the stored partition lengths to the
            new index object.

        Returns
        -------
        ModinIndex
        """
        idx_cache = self._value
        if not callable(idx_cache):
            idx_cache = idx_cache.copy()
        result = ModinIndex(idx_cache, value_ref=self._value_ref)
        result._index_id = self._index_id
        if copy_lengths:
            result._lengths_cache = self._lengths_cache
            result._lengths_id = self._lengths_id
        return result
