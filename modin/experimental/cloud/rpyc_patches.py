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

import numpy
from rpyc.core import netref


def apply_pathes():
    def fixed_make_method(name, doc, orig=netref._make_method):
        if name == "__array__":

            def __array__(self, dtype=None):
                # Note that protocol=-1 will only work between python
                # interpreters of the same version.
                res = netref.pickle.loads(
                    netref.syncreq(
                        self,
                        netref.consts.HANDLE_PICKLE,
                        netref.pickle.HIGHEST_PROTOCOL,
                    )
                )

                if dtype is not None:
                    res = numpy.asarray(res, dtype=dtype)

                return res

            __array__.__doc__ = doc
            return __array__
        return orig(name, doc)

    netref._make_method = fixed_make_method
