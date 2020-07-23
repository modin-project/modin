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

_LOCAL_ATTRS = frozenset(("__new__", "__dict__"))
_WRAP_ATTRS = ("__wrapper_local__", "__wrapper_remote__")


class MetaComparer(type):
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

    def __instancecheck__(self, instance):
        # see if self is a type that has __real_cls__ defined,
        # as if it's a dual-nature class we need to use its internal class to compare
        my_cls = self.__dict__.get("__real_cls__", self)
        try:
            # see if it's a proxying wrapper, in which case
            # use wrapped local class as a comparison base
            my_cls = object.__getattribute__(my_cls, "__wrapper_local__")
        except AttributeError:
            pass
        return issubclass(instance.__class__, my_cls)


class RemoteMeta(MetaComparer):
    """
    Metaclass that relays getting non-existing attributes from
    a proxying object *CLASS* to a remote end transparently.

    Attributes existing on a proxying object are retrieved locally.
    """

    def __getattribute__(self, name):
        if name in _LOCAL_ATTRS:
            # never proxy special attributes, always get them from the class type
            res = object.__getattribute__(self, name)
        else:
            try:
                # Go for proxying class-level attributes first;
                # make sure to check for attribute in self.__dict__ to get the class-level
                # attribute from the class itself, not from some of its parent classes.
                # Also note we use object.__getattribute__() to skip any potential class-level __getattr__
                res = object.__getattribute__(self, "__dict__")[name]
            except KeyError:
                attr_ex = None
                for wrap_name in _WRAP_ATTRS:
                    # now try getting an attribute from an overriding object first,
                    # and only if it fails resort to getting from the remote end
                    try:
                        res = getattr(object.__getattribute__(self, wrap_name), name)
                        break
                    except AttributeError as ex:
                        attr_ex = ex
                        continue
                else:
                    raise attr_ex
        try:
            # note that any attribute might be in fact a data descriptor,
            # account for that
            getter = res.__get__
        except AttributeError:
            return res
        return getter(self)


_KNOWN_DUALS = {}


def make_wrapped_class(local_cls, rpyc_wrapper_name):
    def __new__(cls, *a, **kw):
        if cls is local_cls and cls.__real_cls__ is not local_cls:
            return cls.__real_cls__(*a, **kw)
        return super().__new__(cls)

    namespace = {
        "__real_cls__": None,
        "__new__": __new__,
        "__module__": local_cls.__module__,
    }
    result = MetaComparer(local_cls.__name__, (local_cls,), namespace)
    setattr(sys.modules[local_cls.__module__], local_cls.__name__, result)
    _KNOWN_DUALS[local_cls] = result

    def update_class(_):
        if execution_engine.get() == "Cloudray":
            from . import rpyc_proxy

            result.__real_cls__ = getattr(rpyc_proxy, rpyc_wrapper_name)()
        else:
            result.__real_cls__ = result

    execution_engine.subscribe(update_class)
