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

    def patched_netref_getattribute(
        this, name, orig=netref.BaseNetref.__getattribute__
    ):
        """
        Default RPyC behaviour is to defer almost everything to be always obtained
        from remote side. This is almost always correct except when Python behaves
        strangely. For example, when checking for isinstance() or issubclass() it
        gets obj.__bases__ tuple and uses its elements *after* calling a decref
        on the __bases__, because Python assumes that the class type holds
        a reference to __bases__, which isn't true for RPyC proxy classes, so in
        RPyC case the element gets destroyed and undefined behaviour happens.

        So we're patching RPyC netref __getattribute__ to keep a reference
        for certain read-only properties to better emulate local objects.

        Also __array__() implementation works only for numpy arrays, but not other types,
        like scalars (which should become arrays)
        """
        cls = type(this)
        if not hasattr(cls, "__readonly_cache__"):
            type.__setattr__(cls, "__readonly_cache__", {})

        if name in {"__bases__", "__base__", "__mro__"}:
            cache = object.__getattribute__(cls, "__readonly_cache__")
            try:
                return cache[name]
            except KeyError:
                res = cache[name] = orig(this, name)
                return res
        return orig(this, name)

    netref.BaseNetref.__getattribute__ = patched_netref_getattribute
