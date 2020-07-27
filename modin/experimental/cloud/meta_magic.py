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
import inspect
import types

from modin import execution_engine

_LOCAL_ATTRS = frozenset(("__new__", "__dict__", "__wrapper_remote__"))


class RemoteMeta(type):
    """
    Metaclass that relays getting non-existing attributes from
    a proxying object *CLASS* to a remote end transparently.

    Attributes existing on a proxying object are retrieved locally.
    """

    @property
    def __signature__(self):
        """
        Override detection performed by inspect.signature().
        Defining custom __new__() throws off inspect.signature(ClassType)
        as it returns a signature of __new__(), even if said __new__() is defined
        in a parent class.
        """
        # Note that we create an artificial bound method here, as otherwise
        # self.__init__ is an ordinary function, and inspect.signature() shows
        # "self" argument while it should hide it for our purposes.
        # So we make a method bound to class type (it would normally be bound to instance)
        # and pass that to .signature()
        return inspect.signature(types.MethodType(self.__init__, self))

    def __getattribute__(self, name):
        if name in _LOCAL_ATTRS:
            # never proxy special attributes, always get them from the class type
            res = object.__getattribute__(self, name)
        else:
            try:
                # Go for proxying class-level attributes first;
                # make sure to check for attribute in self.__dict__ to get the class-level
                # attribute from the class itself, not from some of its parent classes.
                # Also note we use object.__getattribute__() to skip any potential
                # class-level __getattr__
                res = object.__getattribute__(self, "__dict__")[name]
            except KeyError:
                try:
                    res = object.__getattribute__(self, name)
                except AttributeError:
                    frame = sys._getframe()
                    try:
                        is_inspect = frame.f_back.f_code.co_filename == inspect.__file__
                    except AttributeError:
                        is_inspect = False
                    finally:
                        del frame
                    if is_inspect:
                        # be always-local for inspect.* functions
                        res = super().__getattribute__(name)
                    else:
                        try:
                            remote = object.__getattribute__(
                                object.__getattribute__(self, "__real_cls__"),
                                "__wrapper_remote__",
                            )
                        except AttributeError:
                            # running in local mode, fall back
                            res = super().__getattribute__(name)
                        else:
                            res = getattr(remote, name)
        try:
            # note that any attribute might be in fact a data descriptor,
            # account for that
            getter = res.__get__
        except AttributeError:
            return res
        return getter(None, self)


_KNOWN_DUALS = {}


def make_wrapped_class(local_cls: type, rpyc_wrapper_name: str):
    """
    Replaces given local class in its module with a descendant class
    which has __new__ overridden (a dual-nature class).
    This new class is instantiated differently depending o
     whether this is done in remote context or local.

    In local context we effectively get the same behaviour, but in remote
    context the created class is actually of separate type which
    proxies most requests to a remote end.

    Parameters
    ----------
    local_cls: class
        The class to replace with a dual-nature class
    rpyc_wrapper_name: str
        The function *name* to make a proxy class type.
        Note that this is specifically taken as string to not import
        "rpyc_proxy" module in top-level, as it requires RPyC to be
        installed, and not all users of Modin (even in experimental mode)
        need remote context.
    """
    namespace = {
        "__real_cls__": None,
        "__new__": None,
        "__module__": local_cls.__module__,
    }
    result = RemoteMeta(local_cls.__name__, (local_cls,), namespace)

    def make_new(__class__):
        """
        Define a __new__() with a __class__ that is closure-bound, needed for super() to work
        """

        def __new__(cls, *a, **kw):
            if cls is result and cls.__real_cls__ is not result:
                return cls.__real_cls__(*a, **kw)
            return super().__new__(cls)

        __class__.__new__ = __new__

    make_new(result)
    setattr(sys.modules[local_cls.__module__], local_cls.__name__, result)
    _KNOWN_DUALS[local_cls] = result

    def update_class(_):
        if execution_engine.get() == "Cloudray":
            from . import rpyc_proxy

            result.__real_cls__ = getattr(rpyc_proxy, rpyc_wrapper_name)(result)
        else:
            result.__real_cls__ = result

    execution_engine.subscribe(update_class)
