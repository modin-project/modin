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

"""Contains utility functions for dispatchers."""

import io


class CustomNewlineIterator:
    r"""
    Used to iterate through files in binary mode line by line where newline != b'\n'.

    Parameters
    ----------
    _file : file-like object
        File-like object to iterate over.
    newline : bytes
        Byte or sequence of bytes indicating line endings.
    """

    def __init__(self, _file, newline):
        self.file = _file
        self.newline = newline
        self.bytes_read = self.chunk_size = 0

    def __iter__(self):
        """
        Iterate over lines.

        Yields
        ------
        bytes
            Data from file.
        """
        buffer_size = io.DEFAULT_BUFFER_SIZE
        chunk = self.file.read(buffer_size)
        self.chunk_size = 0
        while chunk:
            self.bytes_read = 0
            self.chunk_size = len(chunk)
            # split remove newline bytes from line
            lines = chunk.split(self.newline)
            for line in lines[:-1]:
                self.bytes_read += len(line) + len(self.newline)
                yield line
            chunk = self.file.read(buffer_size)
            if lines[-1]:
                # last line can be read without newline bytes
                chunk = lines[-1] + chunk

    def seek(self):
        """Change the stream positition to where the last returned line ends."""
        self.file.seek(self.bytes_read - self.chunk_size, 1)
