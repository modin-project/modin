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

_SPECIAL = frozenset(("__new__", "__dict__"))
_WRAP_ATTRS = ("__wrapper_local__", "__wrapper_remote__")


class MetaComparer(type):
    def __instancecheck__(self, instance):
        return issubclass(instance.__class__, self.__dict__.get("__real_cls__", self))


class RemoteMeta(MetaComparer):
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


_KNOWN_DUALS = {}


def make_wrapped_class(local_cls, cls_name, rpyc_wrapper_name):
    from modin import execution_engine

    current_frame = sys._getframe()
    try:
        module_name = current_frame.f_back.f_globals["__name__"]
    except (AttributeError, KeyError):
        module_name = __name__
    finally:
        del current_frame  # avoid reference cycles

    @classmethod
    def _update_engine(cls, _):
        if execution_engine.get() == "Cloudray":
            from . import rpyc_proxy

            cls.__real_cls__ = getattr(rpyc_proxy, rpyc_wrapper_name)()
        else:
            cls.__real_cls__ = local_cls

    def __new__(cls, *a, **kw):
        if cls.__name__ == cls_name:
            return cls.__real_cls__(*a, **kw)
        return local_cls.__new__(cls)

    namespace = {
        "__real_cls__": None,
        "_update_engine": _update_engine,
        "__new__": __new__,
        "__module__": module_name,
    }
    result = MetaComparer(cls_name, (local_cls,), namespace)
    execution_engine.subscribe(result._update_engine)

    _KNOWN_DUALS[local_cls] = result
    return result
