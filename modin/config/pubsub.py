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

import collections
import typing


class TypeDescriptor(typing.NamedTuple):
    decode: typing.Callable[[str], object]
    normalize: typing.Callable[[object], object]
    verify: typing.Callable[[object], bool]
    help: str


class ExactStr(str):
    """
    To be used in type params where no transformations are needed
    """


_TYPE_PARAMS = {
    str: TypeDescriptor(
        decode=lambda value: value.strip().title(),
        normalize=lambda value: value.strip().title(),
        verify=lambda value: True,
        help="a case-insensitive string",
    ),
    ExactStr: TypeDescriptor(
        decode=lambda value: value,
        normalize=lambda value: value,
        verify=lambda value: True,
        help="a string",
    ),
    bool: TypeDescriptor(
        decode=lambda value: value.strip().lower() in {"true", "yes", "1"},
        normalize=bool,
        verify=lambda value: isinstance(value, bool)
        or (
            isinstance(value, str)
            and value.strip().lower() in {"true", "yes", "1", "false", "no", "0"}
        ),
        help="a boolean flag (any of 'true', 'yes' or '1' in case insensitive manner is considered positive)",
    ),
    int: TypeDescriptor(
        decode=lambda value: int(value.strip()),
        normalize=int,
        verify=lambda value: isinstance(value, int)
        or (isinstance(value, str) and value.strip().isdigit()),
        help="an integer value",
    ),
}

# special marker to distinguish unset value from None value
# as someone may want to use None as a real value for a parameter
_UNSET = object()


class Parameter(object):
    """
    Base class describing interface for configuration entities
    """

    choices: typing.Sequence[str] = None
    type = str
    default = None
    is_abstract = True

    @classmethod
    def _get_raw_from_config(cls) -> str:
        """
        The method that really reads the value from config storage,
        be it some config file or environment variable or whatever.

        Raises KeyError if value is absent.
        """
        raise NotImplementedError()

    @classmethod
    def get_help(cls) -> str:
        """
        Generate user-presentable help for the option
        """
        raise NotImplementedError()

    def __init_subclass__(cls, type, abstract=False, **kw):
        assert type in _TYPE_PARAMS, f"Unsupported variable type: {type}"
        cls.type = type
        cls.is_abstract = abstract
        cls._value = _UNSET
        cls._subs = []
        cls._once = collections.defaultdict(list)
        super().__init_subclass__(**kw)

    @classmethod
    def subscribe(cls, callback):
        cls._subs.append(callback)
        callback(cls)

    @classmethod
    def _get_default(cls):
        return cls.default

    @classmethod
    def get(cls):
        if cls._value is _UNSET:
            # get the value from env
            try:
                raw = cls._get_raw_from_config()
            except KeyError:
                cls._value = cls._get_default()
            else:
                if not _TYPE_PARAMS[cls.type].verify(raw):
                    raise ValueError(f"Unsupported raw value: {raw}")
                cls._value = _TYPE_PARAMS[cls.type].decode(raw)
        return cls._value

    @classmethod
    def put(cls, value):
        cls._check_callbacks(cls._put_nocallback(value))

    @classmethod
    def once(cls, onvalue, callback):
        onvalue = _TYPE_PARAMS[cls.type].normalize(onvalue)
        if onvalue == cls.get():
            callback(cls)
        else:
            cls._once[onvalue].append(callback)

    @classmethod
    def _put_nocallback(cls, value):
        if not _TYPE_PARAMS[cls.type].verify(value):
            raise ValueError(f"Unsupported value: {value}")
        value = _TYPE_PARAMS[cls.type].normalize(value)
        oldvalue, cls._value = cls.get(), value
        return oldvalue

    @classmethod
    def _check_callbacks(cls, oldvalue):
        if oldvalue == cls.get():
            return
        for callback in cls._subs:
            callback(cls)
        for callback in cls._once.pop(cls.get(), ()):
            callback(cls)


__all__ = ["Parameter"]
