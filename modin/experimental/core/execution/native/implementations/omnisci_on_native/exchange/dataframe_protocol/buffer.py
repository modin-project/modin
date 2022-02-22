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

import pyarrow as pa

from typing import Tuple
from modin.core.dataframe.base.exchange.dataframe_protocol.utils import DlpackDeviceType
from modin.core.dataframe.base.exchange.dataframe_protocol.dataframe import (
    ProtocolBuffer,
)


class OmnisciProtocolBuffer(ProtocolBuffer):
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

    def __init__(self, buff: pa.Buffer, allow_copy: bool = True) -> None:
        """
        Handle only regular columns (= numpy arrays) for now.
        """
        self._buff = buff

    @property
    def bufsize(self) -> int:
        """
        Buffer size in bytes.
        """
        return self._buff.size

    @property
    def ptr(self) -> int:
        """
        Pointer to start of the buffer as an integer.
        """
        return self._buff.address

    def __dlpack__(self):
        """
        DLPack not implemented in NumPy yet, so leave it out here.

        Produce DLPack capsule (see array API standard).
        Raises:
            - TypeError : if the buffer contains unsupported dtypes.
            - NotImplementedError : if DLPack support is not implemented
        Useful to have to connect to array libraries. Support optional because
        it's not completely trivial to implement for a Python-only library.
        """
        raise NotImplementedError("__dlpack__")

    def __dlpack_device__(self) -> Tuple[DlpackDeviceType, int]:
        """
        Device type and device ID for where the data in the buffer resides.
        Uses device type codes matching DLPack. Enum members are::
            - CPU = 1
            - CUDA = 2
            - CPU_PINNED = 3
            - OPENCL = 4
            - VULKAN = 7
            - METAL = 8
            - VPI = 9
            - ROCM = 10
        Note: must be implemented even if ``__dlpack__`` is not.
        """

        return (DlpackDeviceType.CPU, None)

    def __repr__(self) -> str:
        """
        Return a string representation for a particular ``Buffer``.

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
