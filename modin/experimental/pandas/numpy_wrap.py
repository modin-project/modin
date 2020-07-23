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

_CAUGHT_NUMPY = "numpy" not in sys.modules
try:
    import numpy as real_numpy
except ImportError:
    pass
else:
    import types
    import copyreg
    from modin import execution_engine
    import modin
    import pandas
    import os

    _EXCLUDE_MODULES = [modin, pandas]
    try:
        import rpyc
    except ImportError:
        pass
    else:
        _EXCLUDE_MODULES.append(rpyc)
    _EXCLUDE_PATHS = tuple(
        os.path.dirname(mod.__file__) + os.sep for mod in _EXCLUDE_MODULES
    )

    class InterceptedNumpy(types.ModuleType):
        __own_attrs__ = set(["__own_attrs__"])

        __spec__ = real_numpy.__spec__
        __current_numpy = real_numpy
        __prev_numpy = real_numpy
        __has_to_warn = not _CAUGHT_NUMPY
        __reducers = {}

        def __init__(self):
            self.__own_attrs__ = set(type(self).__dict__.keys())
            execution_engine.subscribe(self.__update_engine)

        def __swap_numpy(self, other_numpy=None):
            self.__current_numpy, self.__prev_numpy = (
                other_numpy or self.__prev_numpy,
                self.__current_numpy,
            )
            if self.__current_numpy is not real_numpy and self.__has_to_warn:
                import warnings

                warnings.warn(
                    "Was not able to intercept all numpy imports. "
                    "To intercept all of these please do 'import modin.experimental.pandas' as early as possible"
                )
                self.__has_to_warn = False

        def __update_engine(self, _):
            if execution_engine.get() == "Cloudray":
                from modin.experimental.cloud import get_connection

                self.__swap_numpy(get_connection().modules["numpy"])
            else:
                self.__swap_numpy()

        def __make_reducer(self, name):
            try:
                reducer = self.__reducers[name]
            except KeyError:

                def reducer(
                    obj,
                    obj_name=name,
                    real_obj_reducer=getattr(real_numpy, name).__reduce__,
                ):
                    return (
                        getattr(self.__current_numpy, obj_name),
                    ) + real_obj_reducer(obj)[1:]

                self.__reducers[name] = reducer
            return reducer

        def __get_numpy(self):
            frame = sys._getframe()
            try:
                caller_file = frame.f_back.f_back.f_code.co_filename
            except AttributeError:
                return self.__current_numpy
            finally:
                del frame
            if any(caller_file.startswith(p) for p in _EXCLUDE_PATHS):
                return real_numpy
            return self.__current_numpy

        def __getattr__(self, name):
            obj = getattr(self.__get_numpy(), name)
            if isinstance(obj, type):
                copyreg.pickle(obj, self.__make_reducer(name))
            return obj

        def __setattr__(self, name, value):
            if name in self.__own_attrs__:
                super().__setattr__(name, value)
            else:
                setattr(self.__get_numpy(), name, value)

        def __delattr__(self, name):
            if name not in self.__own_attrs__:
                delattr(self.__get_numpy(), name)

    sys.modules["numpy"] = InterceptedNumpy()
