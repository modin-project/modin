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

from . import get_connection
from rpyc.utils.classic import deliver
import rpyc


class WrappingConnection(rpyc.Connection):
    def _netref_factory(self, id_pack):
        result = super()._netref_factory(id_pack)
        real_class = getattr(getattr(result, "__class__", None), "__real_cls__", None)
        return real_class.from_remote_end(result) if real_class else result


class WrappingService(rpyc.ClassicService):
    _protocol = WrappingConnection


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
    class ProxyMeta(type):
        def __repr__(self):
            return f"<proxy for {origin_cls.__module__}.{origin_cls.__name__}:{cls_name or origin_cls.__name__}"

        def __prepare__(self, *args, **kw):
            namespace = type.__prepare__(*args, **kw)
            for entry in dir(origin_cls):
                if entry.startswith("__") and entry.endswith("__"):
                    origin_entry = getattr(origin_cls, entry, None)
                    if callable(origin_entry) and origin_entry != getattr(
                        object, entry, None
                    ):
                        try:
                            remote_entry = getattr(remote_cls, entry)
                        except AttributeError:
                            continue
                        namespace[entry] = remote_entry
            return namespace

    class Wrapper(override, metaclass=ProxyMeta):
        def __init__(self, *a, __remote_end__=None, **kw):
            if __remote_end__ is None:
                __remote_end__ = remote_cls(*a, **kw)
            object.__setattr__(self, "__remote_end__", __remote_end__)

        @classmethod
        def from_remote_end(cls, remote_inst):
            return cls(__remote_end__=remote_inst)

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


def _deliveringWrapper(origin_cls, methods, mixin=None, target_name=None):
    conn = get_connection()
    remote_cls = getattr(conn.modules[origin_cls.__module__], origin_cls.__name__)

    if mixin is None:

        class DeliveringMixin:
            pass

        mixin = DeliveringMixin

    for method in methods:

        def wrapper(self, *args, __remote_conn__=conn, __method_name__=method, **kw):
            args, kw = deliver(__remote_conn__, (args, kw))
            return getattr(self.__remote_end__, __method_name__)(*args, **kw)

        wrapper.__name__ = method
        setattr(mixin, method, wrapper)
    return make_proxy_cls(
        remote_cls, origin_cls, mixin, target_name or origin_cls.__name__
    )


def _prepare_loc_mixin():
    from modin.pandas.indexing import _LocIndexer, _iLocIndexer

    DeliveringLocIndexer = _deliveringWrapper(
        _LocIndexer, ["__getitem__", "__setitem__"]
    )
    DeliveringILocIndexer = _deliveringWrapper(
        _iLocIndexer, ["__getitem__", "__setitem__"]
    )

    class DeliveringMixin:
        @property
        def loc(self):
            return DeliveringLocIndexer(self)

        @property
        def iloc(self):
            return DeliveringILocIndexer(self)

    return DeliveringMixin


def make_dataframe_wrapper():
    from modin.pandas.dataframe import _DataFrame

    DeliveringDataFrame = _deliveringWrapper(
        _DataFrame, ["groupby", "agg", "aggregate"], _prepare_loc_mixin(), "DataFrame"
    )
    return DeliveringDataFrame


def make_base_dataset_wrapper():
    from modin.pandas.base import _BasePandasDataset

    DeliveringBasePandasDataset = _deliveringWrapper(
        _BasePandasDataset,
        ["agg", "aggregate"],
        _prepare_loc_mixin(),
        "BasePandasDataset",
    )
    return DeliveringBasePandasDataset
