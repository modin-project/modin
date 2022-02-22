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

"""
Dataframe exchange protocol implementation.

See more in https://data-apis.org/dataframe-protocol/latest/index.html.
"""

from abc import ABC, abstractmethod
from typing import Optional, Iterable, Sequence

from .column import ProtocolColumn


class ProtocolDataframe(ABC):
    """
    A data frame class, with only the methods required by the interchange protocol defined.

    Instances of this (private) class are returned from
    ``modin.core.dataframe.base.dataframe.dataframe.ModinDataframe.__dataframe__``
    as objects with the methods and attributes defined on this class.

    A "data frame" represents an ordered collection of named columns.
    A column's "name" must be a unique string. Columns may be accessed by name or by position.
    This could be a public data frame class, or an object with the methods and
    attributes defined on this ProtocolDataframe class could be returned from the
    ``__dataframe__`` method of a public data frame class in a library adhering
    to the dataframe interchange protocol specification.
    """

    @property
    @abstractmethod
    def metadata(self):
        """
        Get the metadata for the data frame, as a dictionary with string keys.

        The contents of `metadata` may be anything, they are meant for a library
        to store information that it needs to, e.g., roundtrip losslessly or
        for two implementations to share data that is not (yet) part of the
        interchange protocol specification. For avoiding collisions with other
        entries, please add name the keys with the name of the library
        followed by a period and the desired name, e.g, ``pandas.indexcol``.

        Returns
        -------
        dict
        """
        pass

    @abstractmethod
    def num_columns(self) -> int:
        """
        Return the number of columns in the ProtocolDataframe.

        Returns
        -------
        int
            The number of columns in the ProtocolDataframe.
        """
        pass

    @abstractmethod
    def num_rows(self) -> int:
        """
        Return the number of rows in the ProtocolDataframe, if available.

        Returns
        -------
        int
            The number of rows in the ProtocolDataframe.
        """
        pass

    @abstractmethod
    def num_chunks(self) -> int:
        """
        Return the number of chunks the ProtocolDataframe consists of.

        Returns
        -------
        int
            The number of chunks the ProtocolDataframe consists of.
        """
        pass

    @abstractmethod
    def column_names(self) -> Iterable[str]:
        """
        Return an iterator yielding the column names.

        Yields
        ------
        str
            The name of the column(s).
        """
        pass

    @abstractmethod
    def get_column(self, i: int) -> ProtocolColumn:
        """
        Return the column at the indicated position.

        Parameters
        ----------
        i : int
            Positional index of the column to be returned.

        Returns
        -------
        Column
            The column at the indicated position.
        """
        pass

    @abstractmethod
    def get_column_by_name(self, name: str) -> ProtocolColumn:
        """
        Return the column whose name is the indicated name.

        Parameters
        ----------
        name : str
            String label of the column to be returned.

        Returns
        -------
        Column
            The column whose name is the indicated name.
        """
        pass

    @abstractmethod
    def get_columns(self) -> Iterable[ProtocolColumn]:
        """
        Return an iterator yielding the columns.

        Yields
        ------
        Column
            The ``Column`` object(s).
        """
        pass

    @abstractmethod
    def select_columns(self, indices: Sequence[int]) -> "ProtocolDataframe":
        """
        Create a new ProtocolDataframe by selecting a subset of columns by index.

        Parameters
        ----------
        indices : Sequence[int]
            Column indices to be selected out of the ProtocolDataframe.

        Returns
        -------
        ProtocolDataframe
            A new ProtocolDataframe with selected a subset of columns by index.
        """
        pass

    @abstractmethod
    def select_columns_by_name(self, names: Sequence[str]) -> "ProtocolDataframe":
        """
        Create a new ProtocolDataframe by selecting a subset of columns by name.

        Parameters
        ----------
        names : Sequence[str]
            Column names to be selected out of the ProtocolDataframe.

        Returns
        -------
        ProtocolDataframe
            A new ProtocolDataframe with selected a subset of columns by name.
        """
        pass

    @abstractmethod
    def get_chunks(
        self, n_chunks: Optional[int] = None
    ) -> Iterable["ProtocolDataframe"]:
        """
        Return an iterator yielding the chunks.

        By default `n_chunks=None`, yields the chunks that the data is stored as by the producer.
        If given, `n_chunks` must be a multiple of `self.num_chunks()`,
        meaning the producer must subdivide each chunk before yielding it.

        Parameters
        ----------
        n_chunks : int, optional
            Number of chunks to yield.

        Yields
        ------
        ProtocolDataframe
            A ``ProtocolDataframe`` object(s).
        """
        pass
