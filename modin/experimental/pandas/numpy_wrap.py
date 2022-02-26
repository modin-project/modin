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
The module replaces real NumPy from future "import numpy" statements.

Replacement occurs with a wrapping module that serves attributes from either
local or "remote" NumPy depending on active execution context.
"""

import sys

_CAUGHT_NUMPY = "numpy" not in sys.modules
try:
    import numpy as real_numpy
except ImportError:
    pass
else:
    import types
    import copyreg
    from modin.config import Engine
    from modin.core.execution.dispatching.factories import REMOTE_ENGINES
    import modin
    import pandas
    import os

    _EXCLUDE_MODULES = [modin, pandas, real_numpy]
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
        """
        The class is intended to replace the "numpy" module as seen by outer world.

        Replacement occurs by getting attributes from either local NumPy or remote one when remote context
        is activated.
        It also registers helpers for pickling local NumPy objects in remote context
        and vice versa.

        Attributes
        ----------
        __own_attrs__ : set
            Attributes that are defined in this class so access to them must never be proxied.
        __current_numpy : ModuleType
            The module to which getting NumPy attributes redirects. For example,
            NumPy on remote machine.
        __prev_numpy : ModuleType
            The previous module that was accessed to get the NumPy attributes.
        __has_to_warn : bool
            Determines the situation when it is necessary to give a warning.
        __reducers : dict
            Custom routines that Pickle calls to serialize an instance of a class.
        """

        __own_attrs__ = set(["__own_attrs__"])

        __spec__ = real_numpy.__spec__
        __current_numpy = real_numpy
        __prev_numpy = real_numpy
        __has_to_warn = not _CAUGHT_NUMPY
        __reducers = {}

        def __init__(self):
            self.__own_attrs__ = set(type(self).__dict__.keys())
            Engine.subscribe(self.__update_engine)

        def __swap_numpy(self, other_numpy=None):
            self.__current_numpy, self.__prev_numpy = (
                other_numpy or self.__prev_numpy,
                self.__current_numpy,
            )
            if self.__current_numpy is not real_numpy and self.__has_to_warn:
                import warnings

                warnings.warn(
                    "Was not able to intercept all numpy imports. "
                    + "To intercept all of these please do 'import modin.experimental.pandas' as early as possible"
                )
                self.__has_to_warn = False

        def __update_engine(self, _):
            if Engine.get() in REMOTE_ENGINES:
                from modin.experimental.cloud import get_connection

                self.__swap_numpy(get_connection().modules["numpy"])
            else:
                self.__swap_numpy()

        def __make_reducer(self, name):
            """
            Prepare a "reducer" routine - the one Pickle calls to serialize an instance of a class.

            Note that we need this to allow pickling a local numpy object in "remote numpy" context,
            because without a custom reduce callback pickle complains that what it reduced has a
            different "numpy" class than original.
            """
            try:
                reducer = self.__reducers[name]
            except KeyError:

                def reducer(
                    obj,
                    real_obj=getattr(real_numpy, name),
                    real_obj_reducer=getattr(real_numpy, name).__reduce__,
                ):
                    # See details on __reduce__ protocol in Python docs:
                    # https://docs.python.org/3.6/library/pickle.html#object.__reduce__
                    reduced = real_obj_reducer(obj)
                    if not isinstance(reduced, tuple):
                        return reduced
                    assert isinstance(
                        reduced[0],
                        (type, types.FunctionType, types.BuiltinFunctionType),
                    ), "Do not know how to support this reconstructor"

                    modobj = self.__current_numpy
                    for submod in reduced[0].__module__.split(".")[1:]:
                        modobj = getattr(modobj, submod)
                    reconstruct = getattr(modobj, reduced[0].__name__)
                    # TODO: see if replacing all "real numpy" things in reduced[1:] is needed
                    return (reconstruct,) + reduced[1:]

                self.__reducers[name] = reducer
            return reducer

        def __get_numpy(self):
            frame = sys._getframe()
            try:
                # get the path to module where caller of caller is defined;
                # this function is expected to be called from one of
                # __getattr__, __setattr__ or __delattr__, so this
                # "caller_file" should point to the file that wants a
                # numpy attribute; we want to always give local numpy
                # to modin, numpy and rpyc as it's all internal for us
                caller_file = frame.f_back.f_back.f_code.co_filename
            except AttributeError:
                return self.__current_numpy
            finally:
                del frame
            if any(caller_file.startswith(mod_path) for mod_path in _EXCLUDE_PATHS):
                return real_numpy
            return self.__current_numpy

        def __getattr__(self, name):  # noqa: D105
            # note that __getattr__ is not symmetric to __setattr__, as it is
            # only called when an attribute is not found by usual lookups
            obj = getattr(self.__get_numpy(), name)
            if isinstance(obj, type):
                # register a special callback for pickling
                copyreg.pickle(obj, self.__make_reducer(name))
            return obj

        def __setattr__(self, name, value):  # noqa: D105
            # set our own attributes on the self instance, but pass through
            # setting other attributes to numpy being wrapped
            if name in self.__own_attrs__:
                super().__setattr__(name, value)
            else:
                setattr(self.__get_numpy(), name, value)

        def __delattr__(self, name):  # noqa: D105
            # do not allow to delete our own attributes, pass through
            # deletion of others to numpy being wrapped
            if name not in self.__own_attrs__:
                delattr(self.__get_numpy(), name)

    sys.modules["numpy"] = InterceptedNumpy()
