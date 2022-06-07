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

from modin.config import Engine
from modin.core.execution.dispatching.factories import REMOTE_ENGINES

# the attributes that must be alwasy taken from a local part of dual-nature class,
# never going to remote end
_LOCAL_ATTRS = frozenset(
    (
        "__new__",
        "__dict__",
        "__wrapper_remote__",
        "__real_cls__",
        "__mro__",
        "__class__",
    )
)


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
            return super().__getattribute__(name)
        else:
            try:
                # Go for proxying class-level attributes first;
                # make sure to check for attribute in self.__dict__ to get the class-level
                # attribute from the class itself, not from some of its parent classes.
                res = super().__getattribute__("__dict__")[name]
            except KeyError:
                # Class-level attribute not found in the class itself; it might be present
                # in its parents, but we must first see if we should go to a remote
                # end, because in "remote context" local attributes are only those which
                # are explicitly allowed by being defined in the class itself.
                frame = sys._getframe()
                try:
                    is_inspect = frame.f_back.f_code.co_filename == inspect.__file__
                except AttributeError:
                    is_inspect = False
                finally:
                    del frame
                if is_inspect:
                    # be always-local for inspect.* functions
                    return super().__getattribute__(name)
                else:
                    try:
                        remote = self.__real_cls__.__wrapper_remote__
                    except AttributeError:
                        # running in local mode, fall back
                        return super().__getattribute__(name)
                    return getattr(remote, name)
            else:
                try:
                    # note that any attribute might be in fact a data descriptor,
                    # account for that; we only need it for attributes we get from __dict__[],
                    # because other cases are handled by super().__getattribute__ for us
                    getter = res.__get__
                except AttributeError:
                    return res
                return getter(None, self)


_KNOWN_DUALS = {}


def make_wrapped_class(local_cls: type, rpyc_wrapper_name: str):
    """
    Replaces given local class in its module with a replacement class
    which has __new__ defined (a dual-nature class).
    This new class is instantiated differently depending on
    whether this is done in remote or local context.

    In local context we effectively get the same behavior, but in remote
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
    # get a copy of local_cls attributes' dict but skip _very_ special attributes,
    # because copying them to a different type leads to them not working.
    # Python should create new descriptors automatically for us instead.
    namespace = {
        name: value
        for name, value in local_cls.__dict__.items()
        if not isinstance(value, types.GetSetDescriptorType)
    }
    namespace["__real_cls__"] = None
    namespace["__new__"] = None
    # define a new class the same way original was defined but with replaced
    # metaclass and a few more attributes in namespace
    result = RemoteMeta(local_cls.__name__, local_cls.__bases__, namespace)

    def make_new(__class__):
        """
        Define a __new__() with a __class__ that is closure-bound, needed for super() to work
        """
        # update '__class__' magic closure value - used by super()
        for attr in __class__.__dict__.values():
            if not callable(attr):
                continue
            cells = getattr(attr, "__closure__", None) or ()
            for cell in cells:
                if cell.cell_contents is local_cls:
                    cell.cell_contents = __class__

        def __new__(cls, *a, **kw):
            if cls is result and cls.__real_cls__ is not result:
                return cls.__real_cls__(*a, **kw)
            return super().__new__(cls)

        __class__.__new__ = __new__

    make_new(result)
    setattr(sys.modules[local_cls.__module__], local_cls.__name__, result)
    _KNOWN_DUALS[local_cls] = result

    def update_class(_):
        if Engine.get() in REMOTE_ENGINES:
            from . import rpyc_proxy

            result.__real_cls__ = getattr(rpyc_proxy, rpyc_wrapper_name)(result)
        else:
            result.__real_cls__ = result

    Engine.subscribe(update_class)
