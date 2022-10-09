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
from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, TypedDict

from .utils import ColumnNullType, DlpackDeviceType, DTypeKind


class ColumnBuffers(TypedDict):  # noqa: GL08
    # first element is a buffer containing the column data;
    # second element is the data buffer's associated dtype
    data: Tuple["ProtocolBuffer", Any]

    # first element is a buffer containing mask values indicating missing data;
    # second element is the mask value buffer's associated dtype.
    # None if the null representation is not a bit or byte mask
    validity: Optional[Tuple["ProtocolBuffer", Any]]

    # first element is a buffer containing the offset values for
    # variable-size binary data (e.g., variable-length strings);
    # second element is the offsets buffer's associated dtype.
    # None if the data buffer does not have an associated offsets buffer
    offsets: Optional[Tuple["ProtocolBuffer", Any]]


class CategoricalDescription(TypedDict):  # noqa: GL08
    # whether the ordering of dictionary indices is semantically meaningful
    is_ordered: bool
    # whether a column-style mapping of categorical values to other objects exists
    is_dictionary: bool
    # None if not a column-style categorical.
    categories: Optional["ProtocolColumn"]


class ProtocolBuffer(ABC):
    """
    Data in the buffer is guaranteed to be contiguous in memory.

    Note that there is no dtype attribute present, a buffer can be thought of
    as simply a block of memory. However, if the column that the buffer is
    attached to has a dtype that's supported by DLPack and ``__dlpack__`` is
    implemented, then that dtype information will be contained in the return
    value from ``__dlpack__``.

    This distinction is useful to support both (a) data exchange via DLPack on a
    buffer and (b) dtypes like variable-length strings which do not have a
    fixed number of bytes per element.
    """

    @property
    @abstractmethod
    def bufsize(self) -> int:
        """
        Buffer size in bytes.

        Returns
        -------
        int
        """
        pass

    @property
    @abstractmethod
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.

        Returns
        -------
        int
        """
        pass

    @abstractmethod
    def __dlpack__(self) -> Any:
        """
        Produce DLPack capsule (see array API standard).

        DLPack not implemented in NumPy yet, so leave it out here.

        Raises
        ------
        ``TypeError`` if the buffer contains unsupported dtypes.
        ``NotImplementedError`` if DLPack support is not implemented.

        Notes
        -----
        Useful to have to connect to array libraries. Support optional because
        it's not completely trivial to implement for a Python-only library.
        """
        pass

    @abstractmethod
    def __dlpack_device__(self) -> Tuple[DlpackDeviceType, Optional[int]]:
        """
        Device type and device ID for where the data in the buffer resides.

        Uses device type codes matching DLPack. Enum members are:
            - CPU = 1
            - CUDA = 2
            - CPU_PINNED = 3
            - OPENCL = 4
            - VULKAN = 7
            - METAL = 8
            - VPI = 9
            - ROCM = 10

        Returns
        -------
        tuple
            Device type and device ID.

        Notes
        -----
        Must be implemented even if ``__dlpack__`` is not.
        """
        pass


class ProtocolColumn(ABC):
    """
    A column object, with only the methods and properties required by the interchange protocol defined.

    A column can contain one or more chunks. Each chunk can contain up to three
    buffers - a data buffer, a mask buffer (depending on null representation),
    and an offsets buffer (if variable-size binary; e.g., variable-length strings).

    TBD: Arrow has a separate "null" dtype, and has no separate mask concept.
         Instead, it seems to use "children" for both columns with a bit mask,
         and for nested dtypes. Unclear whether this is elegant or confusing.
         This design requires checking the null representation explicitly.
         The Arrow design requires checking:
         1. the ARROW_FLAG_NULLABLE (for sentinel values)
         2. if a column has two children, combined with one of those children
            having a null dtype.
         Making the mask concept explicit seems useful. One null dtype would
         not be enough to cover both bit and byte masks, so that would mean
         even more checking if we did it the Arrow way.
    TBD: there's also the "chunk" concept here, which is implicit in Arrow as
         multiple buffers per array (= column here). Semantically it may make
         sense to have both: chunks were meant for example for lazy evaluation
         of data which doesn't fit in memory, while multiple buffers per column
         could also come from doing a selection operation on a single
         contiguous buffer.
         Given these concepts, one would expect chunks to be all of the same
         size (say a 10,000 row dataframe could have 10 chunks of 1,000 rows),
         while multiple buffers could have data-dependent lengths. Not an issue
         in pandas if one column is backed by a single NumPy array, but in
         Arrow it seems possible.
         Are multiple chunks *and* multiple buffers per column necessary for
         the purposes of this interchange protocol, or must producers either
         reuse the chunk concept for this or copy the data?

    Notes
    -----
    This ProtocolColumn object can only be produced by ``__dataframe__``,
    so doesn't need its own version or ``__column__`` protocol.
    """

    @abstractmethod
    def size(self) -> int:
        """
        Size of the column, in elements.

        Corresponds to `DataFrame.num_rows()` if column is a single chunk;
        equal to size of this current chunk otherwise.

        Is a method rather than a property because it may cause a (potentially
        expensive) computation for some dataframe implementations.

        Returns
        -------
        int
            Size of the column, in elements.
        """
        pass

    @property
    @abstractmethod
    def offset(self) -> int:
        """
        Get the offset of first element.

        May be > 0 if using chunks; for example for a column
        with N chunks of equal size M (only the last chunk may be shorter),
        ``offset = n * M``, ``n = 0 .. N-1``.

        Returns
        -------
        int
            The offset of first element.
        """
        pass

    @property
    @abstractmethod
    def dtype(self) -> Tuple[DTypeKind, int, str, str]:
        """
        Dtype description as a tuple ``(kind, bit-width, format string, endianness)``.

        * Kind : DTypeKind
        * Bit-width : the number of bits as an integer
        * Format string : data type description format string in Apache Arrow C
                        Data Interface format.
        * Endianness : current only native endianness (``=``) is supported

        Returns
        -------
        tuple
            ``(kind, bit-width, format string, endianness)``.

        Notes
        -----
        - Kind specifiers are aligned with DLPack where possible
          (hence the jump to 20, leave enough room for future extension).
        - Masks must be specified as boolean with either bit width 1 (for bit masks)
          or 8 (for byte masks).
        - Dtype width in bits was preferred over bytes
        - Endianness isn't too useful, but included now in case in the future
          we need to support non-native endianness
        - Went with Apache Arrow format strings over NumPy format strings
          because they're more complete from a dataframe perspective
        - Format strings are mostly useful for datetime specification, and for categoricals.
        - For categoricals, the format string describes the type of the categorical
          in the data buffer. In case of a separate encoding of the categorical
          (e.g. an integer to string mapping), this can be derived from ``self.describe_categorical``.
        - Data types not included: complex, Arrow-style null, binary, decimal,
          and nested (list, struct, map, union) dtypes.
        """
        pass

    @property
    @abstractmethod
    def describe_categorical(self) -> CategoricalDescription:
        """
        If the dtype is categorical, there are two options.

        - There are only values in the data buffer.
        - There is a separate non-categorical Column encoding categorical values.

        TBD: are there any other in-memory representations that are needed?

        Returns
        -------
        dict
            Content of returned dict:
            - "is_ordered" : bool, whether the ordering of dictionary indices is
                             semantically meaningful.
            - "is_dictionary" : bool, whether a mapping of
                                categorical values to other objects exists
            - "categories" : Column representing the (implicit) mapping of indices to
                             category values (e.g. an array of cat1, cat2, ...).
                             None if not a dictionary-style categorical.

        Raises
        ------
        ``TypeError`` if the dtype is not categorical.
        """
        pass

    @property
    @abstractmethod
    def describe_null(self) -> Tuple[ColumnNullType, Any]:
        """
        Return the missing value (or "null") representation the column dtype uses.

        Return as a tuple ``(kind, value)``.
        * Kind: ColumnNullType
        * Value : if kind is "sentinel value", the actual value. If kind is a bit
          mask or a byte mask, the value (0 or 1) indicating a missing value. None
          otherwise.

        Returns
        -------
        tuple
            ``(kind, value)``.
        """
        pass

    @property
    @abstractmethod
    def null_count(self) -> int:
        """
        Get number of null elements, if known.

        Returns
        -------
        int

        Notes
        -----
        Arrow uses -1 to indicate "unknown", but None seems cleaner.
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        """
        Get the metadata for the column.

        See `DataFrame.metadata` for more details.

        Returns
        -------
        dict
        """
        pass

    @abstractmethod
    def num_chunks(self) -> int:
        """
        Return the number of chunks the column consists of.

        Returns
        -------
        int
           The number of chunks the column consists of.
        """
        pass

    @abstractmethod
    def get_chunks(self, n_chunks: Optional[int] = None) -> Iterable["ProtocolColumn"]:
        """
        Return an iterator yielding the chunks.

        By default ``n_chunks=None``, yields the chunks that the data is stored as by the producer.
        If given, ``n_chunks`` must be a multiple of ``self.num_chunks()``,
        meaning the producer must subdivide each chunk before yielding it.

        Parameters
        ----------
        n_chunks : int, optional
            Number of chunks to yield.

        Yields
        ------
        DataFrame
            A ``DataFrame`` object(s).

        Raises
        ------
        ``RuntimeError`` if ``n_chunks`` is not a multiple of ``self.num_chunks()``.
        """
        pass

    @abstractmethod
    def get_buffers(self) -> ColumnBuffers:
        """
        Return a dictionary containing the underlying buffers.

        Returns
        -------
        dict
            - "data": a two-element tuple whose first element is a buffer
              containing the data and whose second element is the data buffer's associated dtype.
            - "validity": a two-element tuple whose first element is a buffer
              containing mask values indicating missing data and
              whose second element is the mask value buffer's
              associated dtype. None if the null representation is not a bit or byte mask.
            - "offsets": a two-element tuple whose first element is a buffer
              containing the offset values for variable-size binary data
              (e.g., variable-length strings) and whose second element is the offsets
              buffer's associated dtype. None if the data buffer does not have
              an associated offsets buffer.
        """
        pass


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

    version = 0  # version of the protocol

    @abstractmethod
    def __dataframe__(
        self, nan_as_null: bool = False, allow_copy: bool = True
    ) -> "ProtocolDataframe":
        """
        Construct a new dataframe interchange object, potentially changing the parameters.

        See more about the protocol in https://data-apis.org/dataframe-protocol/latest/index.html.

        Parameters
        ----------
        nan_as_null : bool, default: False
            A keyword intended for the consumer to tell the producer
            to overwrite null values in the data with ``NaN``.
            This currently has no effect; once support for nullable extension
            dtypes is added, this value should be propagated to columns.
        allow_copy : bool, default: True
            A keyword that defines whether or not the library is allowed
            to make a copy of the data. For example, copying data would be necessary
            if a library supports strided buffers, given that this protocol
            specifies contiguous buffers. Currently, if the flag is set to ``False``
            and a copy is needed, a ``RuntimeError`` will be raised.

        Returns
        -------
        ProtocolDataframe
        """
        pass

    @property
    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
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
    def num_rows(self) -> Optional[int]:
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

        Raises
        ------
        ``RuntimeError`` if ``n_chunks`` is not a multiple of ``self.num_chunks()``.
        """
        pass
