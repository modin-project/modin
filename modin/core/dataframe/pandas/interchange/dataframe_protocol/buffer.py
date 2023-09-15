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

Notes
-----
- Interpreting a raw pointer (as in ``Buffer.ptr``) is annoying and unsafe to
  do in pure Python. It's more general but definitely less friendly than having
  ``to_arrow`` and ``to_numpy`` methods. So for the buffers which lack
  ``__dlpack__`` (e.g., because the column dtype isn't supported by DLPack),
  this is worth looking at again.
"""

import enum
from typing import Tuple

import numpy as np

from modin.core.dataframe.base.interchange.dataframe_protocol.dataframe import (
    ProtocolBuffer,
)
from modin.utils import _inherit_docstrings


@_inherit_docstrings(ProtocolBuffer)
class PandasProtocolBuffer(ProtocolBuffer):
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

    Parameters
    ----------
    x : np.ndarray
        Data to be held by ``Buffer``.
    allow_copy : bool, default: True
        A keyword that defines whether or not the library is allowed
        to make a copy of the data. For example, copying data would be necessary
        if a library supports strided buffers, given that this protocol
        specifies contiguous buffers. Currently, if the flag is set to ``False``
        and a copy is needed, a ``RuntimeError`` will be raised.
    """

    def __init__(self, x: np.ndarray, allow_copy: bool = True) -> None:
        if not x.strides == (x.dtype.itemsize,):
            # The protocol does not support strided buffers, so a copy is
            # necessary. If that's not allowed, we need to raise an exception.
            if allow_copy:
                x = x.copy()
            else:
                raise RuntimeError(
                    "Exports cannot be zero-copy in the case "
                    + "of a non-contiguous buffer"
                )

        # Store the numpy array in which the data resides as a private
        # attribute, so we can use it to retrieve the public attributes
        self._x = x

    @property
    def bufsize(self) -> int:
        return self._x.size * self._x.dtype.itemsize

    @property
    def ptr(self) -> int:
        return self._x.__array_interface__["data"][0]

    def __dlpack__(self):
        raise NotImplementedError("__dlpack__")

    def __dlpack_device__(self) -> Tuple[enum.IntEnum, int]:
        class Device(enum.IntEnum):
            CPU = 1

        return (Device.CPU, None)

    def __repr__(self) -> str:
        """
        Return a string representation for a particular ``PandasProtocolBuffer``.

        Returns
        -------
        str
        """
        return (
            "Buffer("
            + str(
                {
                    "bufsize": self.bufsize,
                    "ptr": self.ptr,
                    "device": self.__dlpack_device__()[0].name,
                }
            )
            + ")"
        )
