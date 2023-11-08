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

import sys

from ipykernel import kernelspec

default_make_ipkernel_cmd = kernelspec.make_ipkernel_cmd


def custom_make_ipkernel_cmd(*args, **kwargs):
    """
    Build modified Popen command list for launching an IPython kernel with MPI.

    Parameters
    ----------
    *args : iterable
        Additional positional arguments to be passed in `default_make_ipkernel_cmd`.
    **kwargs : dict
        Additional keyword arguments to be passed in `default_make_ipkernel_cmd`.

    Returns
    -------
    array
        A Popen command list.

    Notes
    -----
    The parameters of the function should be kept in sync with the ones of the original function.
    """
    mpi_arguments = ["mpiexec", "-n", "1"]
    arguments = default_make_ipkernel_cmd(*args, **kwargs)
    return mpi_arguments + arguments


kernelspec.make_ipkernel_cmd = custom_make_ipkernel_cmd

if __name__ == "__main__":
    kernel_name = "python3mpi"
    display_name = "Python 3 (ipykernel) with MPI"
    dest = kernelspec.install(
        kernel_name=kernel_name, display_name=display_name, prefix=sys.prefix
    )
    print(f"Installed kernelspec {kernel_name} in {dest}")  # noqa: T201
