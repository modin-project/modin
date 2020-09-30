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


class Caster(typing.NamedTuple):
    decode: typing.Callable[[str], object]
    normalize: typing.Callable[[object], object] = lambda x: x
    encode: typing.Callable[[object], str] = str


_CASTERS = {
    str: Caster(
        decode=lambda value: value.strip().title(),
        normalize=lambda value: value.strip().title(),
    ),
    bool: Caster(
        decode=lambda value: value.strip().lower() in {"true", "yes", "1"},
        normalize=bool,
    ),
    int: Caster(decode=lambda value: int(value.strip()), normalize=int),
}

_TYPE_HELP = {
    str: "string",
    bool: "boolean flag (any of 'true', 'yes' or '1' in case insensitive manner is considered positive)",
    int: "integer value",
}


class _ValueMeta(type):
    """
    Metaclass is needed to make classmethod property
    """

    @property
    def value(cls):
        if cls._value is None:
            # get the value from env
            try:
                raw = cls._get_raw_from_config()
            except KeyError:
                cls._value = cls.default
            else:
                cls._value = _CASTERS[cls.type].decode(raw)
        return cls._value

    @value.setter
    def value(cls, value):
        cls._check_callbacks(cls._put_nocallback(value))


class Publisher(object, metaclass=_ValueMeta):
    """
    Base class describing interface for configuration entities
    """

    choices: typing.Sequence[str] = None
    type = str
    default = None

    @classmethod
    def _get_raw_from_config(cls) -> str:
        """
        The method that really reads the value from config storage,
        be it some config file or environment variable or whatever.

        Raises KeyError if value is absent.
        """
        raise NotImplementedError()

    @classmethod
    def _get_help(cls) -> str:
        """
        Generate user-presentable help for the option
        """
        raise NotImplementedError()

    def __init_subclass__(cls, type=None, default=None, **kw):
        assert type in _CASTERS, f"Unsupported variable type: {type}"
        cls.type = type
        cls._value = None
        cls._subs = []
        cls._once = collections.defaultdict(list)
        super().__init_subclass__(**kw)

    @classmethod
    def subscribe(cls, callback):
        cls._subs.append(callback)
        callback(cls)

    @classmethod
    def once(cls, onvalue, callback):
        onvalue = _CASTERS[cls.type].normalize(onvalue)
        if onvalue == cls.value:
            callback(cls)
        else:
            cls._once[onvalue].append(callback)

    @classmethod
    def _put_nocallback(cls, value):
        value = _CASTERS[cls.type].normalize(value.title)
        oldvalue, cls.value = cls.value, value
        return oldvalue

    @classmethod
    def _check_callbacks(cls, oldvalue):
        if oldvalue == cls.value:
            return
        for callback in cls._subs:
            callback(cls)
        for callback in cls._once.pop(cls.value, ()):
            callback(cls)
