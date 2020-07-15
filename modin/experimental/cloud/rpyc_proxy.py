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

_SPECIAL = frozenset(("__new__", "__dict__"))
_WRAP_ATTRS = ("__wrapper_local__", "__wrapper_remote__")


class RemoteMeta(type):
    """
    Metaclass that relays getting non-existing attributes from
    a proxying object *CLASS* to a remote end transparently.

    Attributes existing on a proxying object are retrieved locally.
    """

    def __getattribute__(self, name):
        if name in _SPECIAL:
            # never proxy special attributes, always get them from the class type
            res = object.__getattribute__(self, name)
        else:
            try:
                # go for proxying class-level attributes first
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


_SPECIAL_ATTRS = frozenset(["__name__", "__remote_end__"])


def make_proxy_cls(remote_cls, origin_cls, override, cls_name=None):
    class Wrapper(override):
        def __init__(self, *a, **kw):
            object.__setattr__(self, "__remote_end__", remote_cls(*a, **kw))

        def __getattr__(self, name):
            """
            Any attributes not currently known to Wrapper (i.e. not defined here
            or in override class) will be retrieved from the remote end
            """
            return getattr(self.__remote_end__, name)

        if override.__setattr__ == object.__setattr__:
            # no custom attribute setting, define our own relaying to remote end
            def __setattr__(self, name, value):
                if name not in _SPECIAL_ATTRS:
                    setattr(self.__remote_end__, name, value)
                else:
                    object.__setattr__(self, name, value)

        if override.__delattr__ == object.__delattr__:
            # no custom __delattr__, define our own
            def __delattr__(self, name):
                if name not in _SPECIAL_ATTRS:
                    delattr(self.__remote_end__, name)

    class Wrapped(origin_cls, metaclass=RemoteMeta):
        __name__ = cls_name or origin_cls.__name__
        __wrapper_remote__ = remote_cls
        __wrapper_local__ = Wrapper
        __class__ = Wrapper

        def __new__(cls, *a, **kw):
            return Wrapper(*a, **kw)

    Wrapper.__name__ = Wrapped.__name__

    return Wrapped
