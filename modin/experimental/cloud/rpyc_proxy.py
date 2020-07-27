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

import types

import rpyc
from rpyc.lib.compat import pickle
from rpyc.core import netref

from . import get_connection
from .meta_magic import _LOCAL_ATTRS, RemoteMeta, _KNOWN_DUALS


class WrappingConnection(rpyc.Connection):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._remote_pickle_loads = None
        self._remote_cls_cache = {}


    def deliver(self, local_obj):
        """
        More caching version of rpyc.classic.deliver()
        """
        try:
            local_obj = object.__getattribute__(local_obj, "__remote_end__")
        except AttributeError:
            pass
        if isinstance(local_obj, netref.BaseNetref) and local_obj.____conn__ is self:
            return local_obj
        return self._remote_pickle_loads(bytes(pickle.dumps(local_obj)))

    def _netref_factory(self, id_pack):
        id_name, cls_id, inst_id = id_pack
        id_name = str(id_name)
        if id_name.startswith('modin.') and inst_id:
            try:
                cached_cls = self._remote_cls_cache[(id_name, cls_id)]
            except KeyError:
                result = super()._netref_factory(id_pack)
                self._remote_cls_cache[(id_name, cls_id)] = type(result)
            else:
                result = cached_cls(self, id_pack)
        else:
            result = super()._netref_factory(id_pack)
        # try getting __real_cls__ from result.__class__ BUT make sure to
        # NOT get it from some parent class for result.__class__, otherwise
        # multiple wrappings happen

        # we cannot use 'result.__class__' as this could cause a lookup of
        # '__class__' on remote end
        try:
            local_cls = object.__getattribute__(result, "__class__")
        except AttributeError:
            return result

        try:
            # first of all, check if remote object has a known "wrapping" class
            # example: _DataFrame has DataFrame dual-nature wrapper
            local_cls = _KNOWN_DUALS[local_cls]
        except KeyError:
            pass
        try:
            # Try to get local_cls.__real_cls__ but look it up within
            # local_cls.__dict__ to not grab it from any parent class.
            # Also get the __dict__ by using low-level __getattribute__
            # to override any potential __getattr__ callbacks on the class.
            wrapping_cls = object.__getattribute__(local_cls, "__dict__")[
                "__real_cls__"
            ]
        except (AttributeError, KeyError):
            return result
        return wrapping_cls.from_remote_end(result)

    def _box(self, obj):
        while True:
            try:
                obj = object.__getattribute__(obj, "__remote_end__")
            except AttributeError:
                break
        return super()._box(obj)

    def _init_deliver(self):
        self._remote_pickle_loads = self.modules["rpyc.lib.compat"].pickle.loads


class WrappingService(rpyc.ClassicService):
    _protocol = WrappingConnection

    def on_connect(self, conn):
        super().on_connect(conn)
        conn._init_deliver()


def _in_empty_class():
    class Empty:
        pass

    return frozenset(Empty.__dict__.keys())


_EMPTY_CLASS_ATTRS = _in_empty_class()

_PROXY_LOCAL_ATTRS = frozenset(["__name__", "__remote_end__"])
_NO_OVERRIDE = (
    _LOCAL_ATTRS
    | _PROXY_LOCAL_ATTRS
    | rpyc.core.netref.DELETED_ATTRS
    | frozenset(["__getattribute__"])
    | _EMPTY_CLASS_ATTRS
)


def make_proxy_cls(
    remote_cls: netref.BaseNetref,
    origin_cls: type,
    override: type,
    cls_name: str = None,
):
    """
    Makes a new class type which inherits from <origin_cls> (for isinstance() and issubtype()),
    takes methods from <override> as-is and proxy all requests for other members to <remote_cls>.
    Note that origin_cls and remote_cls are assumed to be the same class types, but one is local
    and other is obtained from RPyC.

    Effectively implements subclassing, but without subclassing. This is needed because it is
    impossible to subclass a remote-obtained class, something in the very internals of RPyC bugs out.

    Parameters
    ----------
    remote_cls: netref.BaseNetref
        Type obtained from RPyC connection, expected to mirror origin_cls
    origin_cls: type
        The class to prepare a proxying wrapping for
    override: type
        The mixin providing methods and attributes to overlay on top of remote values and methods.
    cls_name: str, optional
        The name to give to the resulting class.

    Returns
    -------
    type
        New wrapper that takes attributes from override and relays requests to all other
        attributes to remote_cls
    """

    class ProxyMeta(RemoteMeta):
        """
        This metaclass deals with printing a telling repr() to assist in debugging,
        and to actually implement the "subclass without subclassing" thing by
        directly adding references to attributes of "override" and by making proxy methods
        for other functions of origin_cls. Class-level attributes being proxied is managed
        by RemoteMeta parent.

        Do note that we cannot do the same for certain special members like __getitem__
        because CPython for optimization doesn't do a lookup of "type(obj).__getitem__(foo)" when
        "obj[foo]" is called, but it effectively does "type(obj).__dict__['__getitem__'](foo)"
        (but even without checking for __dict__), so all present methods must be declared
        beforehand.
        """

        def __repr__(self):
            return f"<proxy for {origin_cls.__module__}.{origin_cls.__name__}:{cls_name or origin_cls.__name__}>"

        def __prepare__(*args, **kw):
            """
            Cooks the __dict__ of the type being constructed. Takes attributes from <override> as is
            and adds proxying wrappers for other attributes of <origin_cls>.
            This "manual inheritance" is needed for RemoteMeta.__getattribute__ which first looks into
            type(obj).__dict__ (EXCLUDING parent classes) and then goes to proxy type.
            """
            namespace = type.__prepare__(*args, **kw)
            namespace['__remote_methods__'] = {}

            # try computing overridden differently to allow subclassing one override from another
            no_override = set(_NO_OVERRIDE)
            for base in override.__mro__:
                if base == object:
                    continue
                for attr_name, attr_value in base.__dict__.items():
                    if (
                        attr_name not in namespace
                        and attr_name not in no_override
                        and getattr(object, attr_name, None) != attr_value
                    ):
                        namespace[
                            attr_name
                        ] = attr_value  # force-inherit an attribute manually
                        no_override.add(attr_name)

            for base in origin_cls.__mro__:
                if base == object:
                    continue
                # try unwrapping a dual-nature class first
                while True:
                    try:
                        sub_base = object.__getattribute__(base, "__real_cls__")
                    except AttributeError:
                        break
                    if sub_base is base:
                        break
                    base = sub_base
                for name, entry in base.__dict__.items():
                    if (
                        name not in namespace
                        and name not in no_override
                        and isinstance(entry, types.FunctionType)
                    ):

                        def method(_self, *_args, __method_name__=name, **_kw):
                            cache = object.__getattribute__(_self, "__remote_methods__")
                            try:
                                remote = cache[__method_name__]
                            except KeyError:
                                cache[__method_name__] = remote = getattr(remote_cls, __method_name__)
                            return remote(_self.__remote_end__, *_args, **_kw)

                        method.__name__ = name
                        namespace[name] = method
            return namespace

    class Wrapper(override, origin_cls, metaclass=ProxyMeta):
        """
        Subclass origin_cls replacing attributes with what is defined in override while
        relaying requests for all other attributes to remote_cls.
        """

        __name__ = cls_name or origin_cls.__name__
        __wrapper_remote__ = remote_cls

        def __new__(cls, *a, **kw):
            return override.__new__(cls)

        def __init__(self, *a, __remote_end__=None, **kw):
            if __remote_end__ is None:
                __remote_end__ = remote_cls(*a, **kw)
            while True:
                # unwrap the object if it's a wrapper
                try:
                    __remote_end__ = object.__getattribute__(
                        __remote_end__, "__remote_end__"
                    )
                except AttributeError:
                    break
            object.__setattr__(self, "__remote_end__", __remote_end__)

        @classmethod
        def from_remote_end(cls, remote_inst):
            return cls(__remote_end__=remote_inst)

        def __getattribute__(self, name):
            """
            Implement "default" resolution order to override whatever __getattribute__
            a parent being wrapped may have defined, but only look up on own __dict__
            without looking into ancestors' ones, because we copy them in __prepare__.

            Effectively, any attributes not currently known to Wrapper (i.e. not defined here
            or in override class) will be retrieved from the remote end.

            Algorithm (mimicking default Python behaviour):
            1) check if type(self).__dict__[name] exists and is a get/set data descriptor
            2) check if self.__dict__[name] exists
            3) check if type(self).__dict__[name] is a non-data descriptor
            4) check if type(self).__dict__[name] exists
            5) pass through to remote end
            """
            dct = object.__getattribute__(self, "__dict__")
            if name == "__dict__":
                return dct
            cls_dct = object.__getattribute__(type(self), "__dict__")
            try:
                cls_attr, has_cls_attr = cls_dct[name], True
            except KeyError:
                has_cls_attr = False
            else:
                oget = None
                try:
                    oget = object.__getattribute__(cls_attr, "__get__")
                    object.__getattribute__(cls_attr, "__set__")
                except AttributeError:
                    pass  # not a get/set data descriptor, go next
                else:
                    return oget(self, type(self))
            # type(self).name is not a get/set data descriptor
            try:
                return dct[name]
            except KeyError:
                # instance doesn't have an attribute
                if has_cls_attr:
                    # type(self) has this attribute, but it's not a get/set descriptor
                    if oget:
                        # this attribute is a get data descriptor
                        return oget(self, type(self))
                    return cls_attr  # not a data descriptor whatsoever

            # this instance/class does not have this attribute, pass it through to remote end
            return getattr(dct["__remote_end__"], name)

        if override.__setattr__ == object.__setattr__:
            # no custom attribute setting, define our own relaying to remote end
            def __setattr__(self, name, value):
                if name not in _PROXY_LOCAL_ATTRS:
                    setattr(self.__remote_end__, name, value)
                else:
                    object.__setattr__(self, name, value)

        if override.__delattr__ == object.__delattr__:
            # no custom __delattr__, define our own
            def __delattr__(self, name):
                if name not in _PROXY_LOCAL_ATTRS:
                    delattr(self.__remote_end__, name)

    return Wrapper


def _deliveringWrapper(
    origin_cls: type, methods=(), mixin: type = None, target_name: str = None
):
    """
    Prepare a proxying wrapper for origin_cls which overrides methods specified in
    "methods" with "delivering" versions of methods.
    A "delivering" method is a method which delivers its arguments to a remote end
    before calling the remote method, effectively calling it with arguments passed
    by value, not by reference.
    This is mostly a workaround for RPyC bug when it translates a non-callable
    type to a remote type which has __call__() method (which would raise TypeError
    when called because local class is not callable).

    Note: this could lead to some weird side-effects if any arguments passed
    in are very funny, but this should never happen in a real data science life.

    Parameters
    ----------
    origin_cls: type
        Local class to make a "delivering wrapper" for.
    methods: sequence of method names, optional
        List of methods to override making "delivering wrappers" for.
    mixin: type, optional
        Parent mixin class to subclass (to inherit already prepared wrappers).
        If not specified, a new mixin is created.
    target_name: str, optional
        Name to give to prepared wrapper class.
        If not specified, take the name of local class being wrapped.

    Returns
    -------
    type
        The "delivering wrapper" mixin, to be used in conjunction with make_proxy_cls()
    """
    conn = get_connection()
    remote_cls = getattr(conn.modules[origin_cls.__module__], origin_cls.__name__)

    if mixin is None:

        class DeliveringMixin:
            pass

        mixin = DeliveringMixin

    for method in methods:

        def wrapper(self, *args, __remote_conn__=conn, __method_name__=method, **kw):
            args = tuple(__remote_conn__.deliver(x) for x in args)
            kw = {k: __remote_conn__.deliver(v) for k, v in kw.items()}
            cache = object.__getattribute__(self, '__remote_methods__')
            try:
                remote = cache[__method_name__]
            except KeyError:
                cache[__method_name__] = remote = getattr(remote_cls, __method_name__)
            return remote(self.__remote_end__, *args, **kw)

        wrapper.__name__ = method
        setattr(mixin, method, wrapper)
    return make_proxy_cls(
        remote_cls, origin_cls, mixin, target_name or origin_cls.__name__
    )


def _prepare_loc_mixin():
    """
    Prepare a mixin that overrides .loc and .iloc properties with versions
    which return a special "delivering" instances of indexers.
    """
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
            return DeliveringLocIndexer(self.__remote_end__)

        @property
        def iloc(self):
            return DeliveringILocIndexer(self.__remote_end__)

    return DeliveringMixin


def make_dataframe_wrapper(DataFrame):
    """
    Prepares a "delivering wrapper" proxy class for DataFrame.
    It makes DF.loc, DF.groupby() and other methods listed below deliver their
    arguments to remote end by value.
    """
    DeliveringDataFrame = _deliveringWrapper(
        DataFrame,
        ["groupby", "agg", "aggregate", "__getitem__", "astype", "drop", "merge"],
        _prepare_loc_mixin(),
        "DataFrame",
    )
    return DeliveringDataFrame


def make_base_dataset_wrapper(BasePandasDataset):
    """
    Prepares a "delivering wrapper" proxy class for BasePandasDataset.
    Look for deatils in make_dataframe_wrapper() and _deliveringWrapper().
    """
    DeliveringBasePandasDataset = _deliveringWrapper(
        BasePandasDataset,
        ["agg", "aggregate"],
        _prepare_loc_mixin(),
        "BasePandasDataset",
    )
    return DeliveringBasePandasDataset


def make_dataframe_groupby_wrapper(DataFrameGroupBy):
    """
    Prepares a "delivering wrapper" proxy class for DataFrameGroupBy.
    Look for deatils in make_dataframe_wrapper() and _deliveringWrapper().
    """
    DeliveringDataFrameGroupBy = _deliveringWrapper(
        DataFrameGroupBy, ["agg", "aggregate", "apply"], target_name="DataFrameGroupBy",
    )
    return DeliveringDataFrameGroupBy


def make_series_wrapper(Series):
    """
    Prepares a "delivering wrapper" proxy class for Series.
    Note that for now _no_ methods that really deliver their arguments by value
    are overridded here, so what it mostly does is it produces a wrapper class
    inherited from normal Series but wrapping all access to remote end transparently.
    """
    return _deliveringWrapper(Series, target_name="Series")
