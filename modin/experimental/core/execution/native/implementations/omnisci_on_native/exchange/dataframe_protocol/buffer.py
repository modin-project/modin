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

"""The module houses OmnisciOnNative implementation of the Buffer class of DataFrame exchange protocol."""

import pyarrow as pa
from typing import Tuple, Optional

from modin.core.dataframe.base.exchange.dataframe_protocol.utils import DlpackDeviceType
from modin.core.dataframe.base.exchange.dataframe_protocol.dataframe import (
    ProtocolBuffer,
)
from modin.utils import _inherit_docstrings


@_inherit_docstrings(ProtocolBuffer)
class OmnisciProtocolBuffer(ProtocolBuffer):
    """
    Wrapper of the ``pyarrow.Buffer`` object representing a continuous segment of memory.

    Parameters
    ----------
    buff : pyarrow.Buffer
        Data to be held by ``Buffer``.
    size : int, optional
        Size of the buffer in bytes, if not specified use ``buff.size``.
        The parameter may be usefull for specifying the size of a virtual chunk.
    """

    def __init__(self, buff: pa.Buffer, size: Optional[int] = None) -> None:
        self._buff = buff
        self._size = self._buff.size if size is None else size

    @property
    def bufsize(self) -> int:
        return self._size

    @property
    def ptr(self) -> int:
        return self._buff.address

    def __dlpack__(self):
        raise NotImplementedError("__dlpack__")

    def __dlpack_device__(self) -> Tuple[DlpackDeviceType, int]:
        return (DlpackDeviceType.CPU, None)

    def __repr__(self) -> str:
        """
        Produce string representation of the buffer.

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
