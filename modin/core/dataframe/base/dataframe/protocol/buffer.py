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
- Interpreting a raw pointer (as in ``ProtocolBuffer.ptr``) is annoying and unsafe to
  do in pure Python. It's more general but definitely less friendly than having
  ``to_arrow`` and ``to_numpy`` methods. So for the buffers which lack
  ``__dlpack__`` (e.g., because the column dtype isn't supported by DLPack),
  this is worth looking at again.
"""

from abc import ABC, abstractmethod
from typing import Tuple

from .utils import DlpackDeviceType


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
    def __dlpack__(self):
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
    def __dlpack_device__(self) -> Tuple[DlpackDeviceType, int]:
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
