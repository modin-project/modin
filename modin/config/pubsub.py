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

"""Module houses ``Parameter`` class - base class for all configs."""

import contextlib
import warnings
from collections import defaultdict
from enum import IntEnum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    DefaultDict,
    Iterator,
    NamedTuple,
    Optional,
    Tuple,
    cast,
)

if TYPE_CHECKING:
    from modin.config.envvars import EnvironmentVariable


class DeprecationDescriptor:
    """
    Describe deprecated parameter.

    Parameters
    ----------
    parameter : type[Parameter]
        Deprecated parameter.
    new_parameter : type[Parameter], optional
        If there's a replacement parameter for the deprecated one, specify it here.
    when_removed : str, optional
        If known, the exact release when the deprecated parameter is planned to be removed.
    """

    _parameter: type["Parameter"]
    _new_parameter: Optional[type["Parameter"]]
    _when_removed: str

    def __init__(
        self,
        parameter: type["Parameter"],
        new_parameter: Optional[type["Parameter"]] = None,
        when_removed: Optional[str] = None,
    ):
        self._parameter = parameter
        self._new_parameter = new_parameter
        self._when_removed = "a future" if when_removed is None else when_removed

    def deprecation_message(self, use_envvar_names: bool = False) -> str:
        """
        Generate a message to be used in a warning raised when using the deprecated parameter.

        Parameters
        ----------
        use_envvar_names : bool, default: False
            Whether to use environment variable names in the warning. If ``True``, both
            ``self._parameter`` and ``self._new_parameter`` have to be a type of ``EnvironmentVariable``.

        Returns
        -------
        str
        """
        name = (
            cast("EnvironmentVariable", self._parameter).varname
            if use_envvar_names
            else self._parameter.__name__
        )
        msg = f"'{name}' is deprecated and will be removed in {self._when_removed} version."
        if self._new_parameter is not None:
            new_name = (
                cast("EnvironmentVariable", self._new_parameter).varname
                if use_envvar_names
                else self._new_parameter.__name__
            )
            msg += f" Use '{new_name}' instead."
        return msg


class TypeDescriptor(NamedTuple):
    """
    Class for config data manipulating of exact type.

    Parameters
    ----------
    decode : callable
        Callable to decode config value from the raw data.
    normalize : callable
        Callable to bring different config value variations to
        the single form.
    verify : callable
        Callable to check that config value satisfies given config
        type requirements.
    help : str
        Class description string.
    """

    decode: Callable[[str], object]
    normalize: Callable[[object], object]
    verify: Callable[[object], bool]
    help: str


class ExactStr(str):
    """Class to be used in type params where no transformations are needed."""


_TYPE_PARAMS = {
    str: TypeDescriptor(
        decode=lambda value: value.strip().title(),
        normalize=lambda value: str(value).strip().title(),
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
        normalize=int,  # type: ignore
        verify=lambda value: isinstance(value, int)
        or (isinstance(value, str) and value.strip().isdigit()),
        help="an integer value",
    ),
    dict: TypeDescriptor(
        decode=lambda value: {
            key: int(val) if val.isdigit() else val
            for key_value in value.split(",")
            for key, val in [[v.strip() for v in key_value.split("=", maxsplit=1)]]
        },
        normalize=lambda value: (
            value
            if isinstance(value, dict)
            else {
                key: int(val) if val.isdigit() else val
                for key_value in str(value).split(",")
                for key, val in [[v.strip() for v in key_value.split("=", maxsplit=1)]]
            }
        ),
        verify=lambda value: isinstance(value, dict)
        or (
            isinstance(value, str)
            and all(
                key_value.find("=") not in (-1, len(key_value) - 1)
                for key_value in value.split(",")
            )
        ),
        help="a sequence of KEY=VALUE values separated by comma (Example: 'KEY1=VALUE1,KEY2=VALUE2,KEY3=VALUE3')",
    ),
}

# special marker to distinguish unset value from None value
# as someone may want to use None as a real value for a parameter
_UNSET = object()


class ValueSource(IntEnum):  # noqa: PR01
    """Class that describes the method of getting the value for a parameter."""

    # got from default, i.e. neither user nor configuration source had the value
    DEFAULT = 0
    # set by user
    SET_BY_USER = 1
    # got from parameter configuration source, like environment variable
    GOT_FROM_CFG_SOURCE = 2


class Parameter(object):
    """
    Base class describing interface for configuration entities.

    Attributes
    ----------
    choices : Optional[Sequence[str]]
        Array with possible options of ``Parameter`` values.
    type : str
        String that denotes ``Parameter`` type.
    default : Optional[Any]
        ``Parameter`` default value.
    is_abstract : bool, default: True
        Whether or not ``Parameter`` is abstract.
    _value_source : Optional[ValueSource]
        Source of the ``Parameter`` value, should be set by
        ``ValueSource``.
    _deprecation_descriptor : Optional[DeprecationDescriptor]
        Indicate whether this parameter is deprecated.
    """

    choices: Optional[Tuple[str, ...]] = None
    type = str
    default: Optional[Any] = None
    is_abstract = True
    _value_source: Optional[ValueSource] = None
    _value: Any = _UNSET
    _subs: list = []
    _once: DefaultDict[Any, list] = defaultdict(list)
    _deprecation_descriptor: Optional[DeprecationDescriptor] = None

    @classmethod
    def _get_raw_from_config(cls) -> str:
        """
        Read the value from config storage.

        Returns
        -------
        str
            Config raw value.

        Raises
        ------
        KeyError
            If value is absent.

        Notes
        -----
        Config storage can be config file or environment variable or whatever.
        Method should be implemented in the child class.
        """
        raise NotImplementedError()

    @classmethod
    def get_help(cls) -> str:
        """
        Generate user-presentable help for the option.

        Returns
        -------
        str

        Notes
        -----
        Method should be implemented in the child class.
        """
        raise NotImplementedError()

    def __init_subclass__(cls, type: Any, abstract: bool = False, **kw: dict):
        """
        Initialize subclass.

        Parameters
        ----------
        type : Any
            Type of the config.
        abstract : bool, default: False
            Whether config is abstract.
        **kw : dict
            Optional arguments for config initialization.
        """
        assert type in _TYPE_PARAMS, f"Unsupported variable type: {type}"
        cls.type = type
        cls.is_abstract = abstract
        cls._value = _UNSET
        cls._subs = []
        cls._once = defaultdict(list)
        super().__init_subclass__(**kw)

    @classmethod
    def subscribe(cls, callback: Callable) -> None:
        """
        Add `callback` to the `_subs` list and then execute it.

        Parameters
        ----------
        callback : callable
            Callable to execute.
        """
        cls._subs.append(callback)
        callback(cls)

    @classmethod
    def _get_default(cls) -> Any:
        """
        Get default value of the config.

        Returns
        -------
        Any
        """
        return cls.default

    @classmethod
    def get_value_source(cls) -> ValueSource:
        """
        Get value source of the config.

        Returns
        -------
        ValueSource
        """
        if cls._value_source is None:
            # dummy call to .get() to initialize the value
            cls.get()
        assert (
            cls._value_source is not None
        ), "_value_source must be initialized by now in get()"
        return cls._value_source

    @classmethod
    def get(cls) -> Any:
        """
        Get config value.

        Returns
        -------
        Any
            Decoded and verified config value.
        """
        if cls._deprecation_descriptor is not None:
            warnings.warn(
                cls._deprecation_descriptor.deprecation_message(), FutureWarning
            )
        if cls._value is _UNSET:
            # get the value from env
            try:
                raw = cls._get_raw_from_config()
            except KeyError:
                cls._value = cls._get_default()
                cls._value_source = ValueSource.DEFAULT
            else:
                if not _TYPE_PARAMS[cls.type].verify(raw):
                    raise ValueError(f"Unsupported raw value: {raw}")
                cls._value = _TYPE_PARAMS[cls.type].decode(raw)
                cls._value_source = ValueSource.GOT_FROM_CFG_SOURCE
        return cls._value

    @classmethod
    def put(cls, value: Any) -> None:
        """
        Set config value.

        Parameters
        ----------
        value : Any
            Config value to set.
        """
        if cls._deprecation_descriptor is not None:
            warnings.warn(
                cls._deprecation_descriptor.deprecation_message(), FutureWarning
            )
        cls._check_callbacks(cls._put_nocallback(value))
        cls._value_source = ValueSource.SET_BY_USER

    @classmethod
    def once(cls, onvalue: Any, callback: Callable) -> None:
        """
        Execute `callback` if config value matches `onvalue` value.

        Otherwise accumulate callbacks associated with the given `onvalue`
        in the `_once` container.

        Parameters
        ----------
        onvalue : Any
            Config value to set.
        callback : callable
            Callable that should be executed if config value matches `onvalue`.
        """
        onvalue = _TYPE_PARAMS[cls.type].normalize(onvalue)
        if onvalue == cls.get():
            callback(cls)
        else:
            cls._once[onvalue].append(callback)

    @classmethod
    def _put_nocallback(cls, value: Any) -> Any:
        """
        Set config value without executing callbacks.

        Parameters
        ----------
        value : Any
            Config value to set.

        Returns
        -------
        Any
            Replaced (old) config value.
        """
        if not _TYPE_PARAMS[cls.type].verify(value):
            raise ValueError(f"Unsupported value: {value}")
        value = _TYPE_PARAMS[cls.type].normalize(value)
        oldvalue, cls._value = cls.get(), value
        return oldvalue

    @classmethod
    def _check_callbacks(cls, oldvalue: Any) -> None:
        """
        Execute all needed callbacks if config value was changed.

        Parameters
        ----------
        oldvalue : Any
            Previous (old) config value.
        """
        if oldvalue == cls.get():
            return
        for callback in cls._subs:
            callback(cls)
        for callback in cls._once.pop(cls.get(), ()):
            callback(cls)

    @classmethod
    def add_option(cls, choice: Any) -> Any:
        """
        Add a new choice for the parameter.

        Parameters
        ----------
        choice : Any
            New choice to add to the available choices.

        Returns
        -------
        Any
            Added choice normalized according to the parameter type.
        """
        if cls.choices is not None:
            if not _TYPE_PARAMS[cls.type].verify(choice):
                raise ValueError(f"Unsupported choice value: {choice}")
            choice = _TYPE_PARAMS[cls.type].normalize(choice)
            if choice not in cls.choices:
                cls.choices += (choice,)
            return choice
        raise TypeError("Cannot add a choice to a parameter where choices is None")


@contextlib.contextmanager
def context(**config: dict[str, Any]) -> Iterator[None]:
    """
    Set a value(s) for the specified config(s) from ``modin.config`` in the scope of the context.

    Parameters
    ----------
    **config : dict[str, Any]
        Keyword describing a name of a config variable from ``modin.config`` as a key
        and a new value as a value.

    Examples
    --------
    >>> RangePartitioning.get()
    False
    >>> with context(RangePartitioning=True):
    ...     print(RangePartitioning.get()) # True
    True
    False
    >>> RangePartitioning.get()
    False
    >>> with context(RangePartitioning=True, AsyncReadMode=True):
    ...     print(RangePartitioning.get()) # True
    ...     print(AsyncReadMode.get()) # True
    True
    True
    >>> RangePartitioning.get()
    False
    >>> AsyncReadMode.get()
    False
    """
    import modin.config as cfg

    old_values = {}
    for name, val in config.items():
        var = getattr(cfg, name)
        old_values[var] = var.get()
        var.put(val)
    try:
        yield
    finally:
        for var, val in old_values.items():
            var.put(val)


__all__ = ["Parameter", "context"]
