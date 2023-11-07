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

"""Module with classes and utilities for deferred remote execution in Ray workers."""
from builtins import NotImplementedError
from enum import Enum
from itertools import islice
from types import GeneratorType
from typing import (
    Any,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Tuple,
    Union,
)

import pandas
import ray
from ray._private.services import get_node_ip_address
from ray.util.client.common import ClientObjectRef

ObjectRefType = Union[ray.ObjectRef, ClientObjectRef, None]
ObjectRefOrListType = Union[ObjectRefType, List[ObjectRefType]]
ListOrTuple = (list, tuple)


class DeferredExecution:
    """
    Deferred execution task.

    This class represents a single node in the execution tree. The input is either
    an object reference or another node on which this node depends.
    The output is calculated by the specified Callable.

    If the input is a DeferredExecution node, it is executed first and the execution
    output is used as the input for this one. All the executions are performed in a
    single batch (i.e. using a single remote call) and the results are saved in all
    the nodes with ref_counter > 1.

    Parameters
    ----------
    data : ObjectRefType or DeferredExecution
        The execution input.
    func : callable or ObjectRefType
        A function to be executed.
    args : iterable
        Additional positional arguments to be passed in `func`.
    kwargs : dict
        Additional keyword arguments to be passed in `func`.
    num_returns : int
        The number of the return values.
    flat_args : bool, optional
        True means that there are no lists or DeferredExecution objects in `args`.
        In this case, no arguments processing is performed and `args` is passed
        to the remote method as is.
    flat_kwargs : bool, optional
        The same as `flat_args` but for the `kwargs` values.
    """

    def __init__(
        self,
        data: Union[
            ObjectRefType,
            "DeferredExecution",
            List[Union[ObjectRefType, "DeferredExecution"]],
        ],
        func: Union[Callable, ObjectRefType],
        args: Iterable[Any],
        kwargs: Dict[str, Any],
        num_returns=1,
        flat_args: Optional[bool] = None,
        flat_kwargs: Optional[bool] = None,
    ):
        ref = DeferredExecution._ref
        if flat_args is None:
            if has_list_or_de(args):
                flat_args = False
                for a in args:
                    ref(a)
            else:
                flat_args = True
        if flat_kwargs is None:
            if has_list_or_de(kwargs.values()):
                flat_kwargs = False
                for a in kwargs.values():
                    ref(a)
            else:
                flat_kwargs = True
        ref(data)
        self.data = data
        self.task = (func, args, kwargs, flat_args, flat_kwargs)
        self.num_returns = num_returns
        self.ref_counter = 0

    @staticmethod
    def _ref(obj):
        """
        Increment the `ref_counter` if `obj` is a `DeferredExecution`.

        Parameters
        ----------
        obj : Any
        """
        if isinstance(obj, DeferredExecution):
            obj.ref_count(1)
        elif isinstance(obj, ListOrTuple):
            ref = DeferredExecution._ref
            for o in obj:
                ref(o)

    def exec(
        self,
    ) -> Tuple[ObjectRefOrListType, Union["MetaList", List], Union[int, List[int]]]:
        """
        Execute this task, if required.

        Returns
        -------
        tuple
            The execution result, MetaList, containing the length, width and
            the worker's ip address (the last value in the list) and the values
            offset in the list. I.e. length = meta_list[offset],
            width = meta_list[offset + 1], ip = meta_list[-1].
        """
        if not self.has_result:
            data = self.data
            if self.num_returns == 1 and not isinstance(data, DeferredExecution):
                task = self.task
                if task[3] and task[4]:
                    result, length, width, ip = remote_exec_func.remote(
                        task[0], data, *task[1], **task[2]
                    )
                    meta = MetaList([length, width, ip])
                    self._set_result(result, meta, 0)
                    return result, meta, 0

            consumers, output = deconstruct(self)
            num_returns = sum(c.num_returns for c in consumers) + 1
            results = _remote_exec_chain(num_returns, *output)
            meta = MetaList(results.pop())
            meta_off = 0
            results = iter(results)
            for de in consumers:
                if de.num_returns == 1:
                    de._set_result(next(results), meta, meta_off)
                    meta_off += 2
                else:
                    res = list(islice(results, num_returns))
                    offsets = list(range(0, 2 * num_returns, 2))
                    de._set_result(res, meta, offsets)
                    meta_off += 2 * num_returns

        return self.data, self.meta, self.meta_off

    def _set_result(
        self,
        result: ObjectRefOrListType,
        meta: "MetaList",
        meta_off: Union[int, List[int]],
    ):
        """
        Set the execution result.

        Parameters
        ----------
        result : ObjectRefOrListType
        meta : MetaList
        meta_off : int or list of int
        """
        del self.task
        self.data = result
        self.meta = meta
        self.meta_off = meta_off

    @property
    def has_result(self):
        """
        Return true if this task has already been executed and the result is set.

        Returns
        -------
        bool
        """
        return not hasattr(self, "task")

    def ref_count(self, diff: int):
        """
        Increment the `ref_counter`.

        Parameters
        ----------
        diff : int
        """
        self.ref_counter += diff
        assert self.ref_counter >= 0

    def __reduce__(self):
        """Not serializable."""
        raise NotImplementedError()


class MetaList:
    """
    Meta information, containing the result lengths and the worker address.

    Parameters
    ----------
    obj : ray.ObjectID or list
    """

    def __init__(self, obj: Union[ray.ObjectID, ClientObjectRef, List]):
        self._obj = obj

    def __getitem__(self, index):
        """
        Get item at the specified index.

        Parameters
        ----------
        index : int

        Returns
        -------
        Any
        """
        obj = self._obj
        if not isinstance(obj, list):
            from modin.core.execution.ray.common import RayWrapper

            self._obj = obj = RayWrapper.materialize(obj)
        return obj[index]

    def __setitem__(self, index, value):
        """
        Set item at the specified index.

        Parameters
        ----------
        index : int
        value : Any
        """
        obj = self._obj
        if not isinstance(obj, list):
            from modin.core.execution.ray.common import RayWrapper

            self._obj = obj = RayWrapper.materialize(obj)
        obj[index] = value


class DeferredExecutionException(Exception):
    """
    Execution exception wrapper.

    Parameters
    ----------
    msg : str
    cause : Exception
    """

    def __init__(self, msg: str, cause: Exception):
        super().__init__(msg, cause)


class _Tag(Enum):  # noqa: PR01
    """A set of special values used for the method arguments de/construction."""

    # The next item is an execution chain
    CHAIN = 0
    # The next item is a reference
    REF = 1
    # The next item a list
    LIST = 2
    # End of list or chain
    END = 3


ListOrDe = (DeferredExecution, list, tuple)


def has_list_or_de(it: Iterable):
    """
    Check if the specified iterable contains either a list or a DeferredExecution object.

    Parameters
    ----------
    it : Iterable

    Returns
    -------
    bool
    """
    return any(isinstance(i, ListOrDe) for i in it)


def deconstruct(de: DeferredExecution) -> Tuple[List[DeferredExecution], List[Any]]:
    """
    Convert the specified execution tree to a flat list.

    This is required for the automatic Ray object references
    materialization before passing the list to a Ray worker.

    The format of the list is the following:
    <input object> sequence<<function> <n><args> <n><kwargs> <ref> <res>>...
    If <n> is >= 0, then the next n objects are the function arguments.
    If <n> is -1, it means that the method arguments contain list and/or
    DeferredExecution (chain) objects. In this case the next values are read
    one by one until `_Tag.END` is encountered. If the value is `_Tag.LIST`,
    then the next sequence of values up to `_Tag.END` is converted to list.
    If the value is `_Tag.CHAIN`, then the next sequence of values up to
    `_Tag.END` has exactly the same format, as described here.
    If the value is `_Tag.REF`, then the next value is a reference id, i.e.
    the actual value should be retrieved by this id from the previously
    saved objects. The <input object> could also be `_Tag.REF` or `_Tag.LIST`.

    If <n> before <kwargs> is >=0, then the next 2*n values are the argument
    names and values in the following format - [name1, value1, name2, value2...].
    If it's -1, then the next values are converted to list in the same way as
    <args> and the argument names are the next len(<args>) values.

    <ref> is an integer reference id. If it's not 0, then there is another
    chain referring to the execution result of this method and, thus, it must
    be saved so that other chains could retrieve the object by the id.

    <res> is a 'get result' flag. If it's True, then the method execution
    result must not only be passed to the next method in the chain, but also
    returned to the caller. The object length and width are added to the meta list.

    Parameters
    ----------
    de : DeferredExecution

    Returns
    -------
    list
    """
    stack = []
    result_consumers = []
    output = []
    de.ref_count(1)
    stack.append(_deconstruct_chain(de, output, stack, result_consumers))
    while stack:
        try:
            gen = stack.pop()
            next_gen = next(gen)
            stack.append(gen)
            stack.append(next_gen)
        except StopIteration:
            ...
    return result_consumers, output


def _deconstruct_chain(
    de: DeferredExecution,
    output: List,
    stack: List,
    result_consumers: List[DeferredExecution],
):  # noqa: GL08
    out_append = output.append
    out_extend = output.extend
    while True:
        de.ref_count(-1)
        if (out_pos := getattr(de, "out_pos", None)) and not de.has_result:
            out_append(_Tag.REF)
            out_append(out_pos)
            output[out_pos] = out_pos
            if de.ref_counter == 0:
                output[out_pos + 1] = 0
                result_consumers.remove(de)
            break
        elif not isinstance(data := de.data, DeferredExecution):
            if isinstance(data, ListOrTuple):
                yield _deconstruct_list(
                    data, output, stack, result_consumers, out_append
                )
            else:
                out_append(data)
            if not de.has_result:
                stack.append(de)
            break
        else:
            stack.append(de)
            de = data

    assert stack and isinstance(stack[-1], DeferredExecution)
    while stack and isinstance(stack[-1], DeferredExecution):
        de = stack.pop()
        task = de.task
        args = task[1]
        kwargs: Dict[str, Any] = task[2]
        out_append(task[0])
        if task[3]:
            out_append(len(args))
            out_extend(args)
        else:
            out_append(-1)
            yield _deconstruct_list(args, output, stack, result_consumers, out_append)
        if task[4]:
            out_append(len(kwargs))
            for item in kwargs.items():
                out_extend(item)
        else:
            out_append(-1)
            yield _deconstruct_list(
                kwargs.values(), output, stack, result_consumers, out_append
            )
            out_extend(kwargs)

        out_append(0)  # Placeholder for ref id
        if de.ref_counter > 0:
            de.out_pos = len(output) - 1  # Ref id position
            result_consumers.append(de)
            out_append(1)  # Return result for this node
        else:
            out_append(0)


def _deconstruct_list(
    lst: Iterable,
    output: List,
    stack: List,
    result_consumers: List[DeferredExecution],
    out_append,
):  # noqa: GL08
    for obj in lst:
        if isinstance(obj, DeferredExecution):
            if out_pos := getattr(obj, "out_pos", None):
                obj.ref_count(-1)
                if obj.has_result:
                    if isinstance(obj.data, ListOrTuple):
                        yield _deconstruct_list(
                            obj.data, output, stack, result_consumers, out_append
                        )
                    else:
                        out_append(obj.data)
                else:
                    out_append(_Tag.REF)
                    out_append(out_pos)
                    output[out_pos] = out_pos
                    if obj.ref_counter == 0:
                        output[out_pos + 1] = 0
                        result_consumers.remove(obj)
            else:
                out_append(_Tag.CHAIN)
                yield _deconstruct_chain(obj, output, stack, result_consumers)
                out_append(_Tag.END)
        elif isinstance(obj, ListOrTuple):
            out_append(_Tag.LIST)
            yield _deconstruct_list(obj, output, stack, result_consumers, out_append)
        else:
            out_append(obj)
    out_append(_Tag.END)


def construct(num_returns: int, args: Tuple) -> Generator:  # pragma: no cover
    """
    Construct and execute the specified chain.

    This function is called in a worker process. The last value, returned by
    this generator, is the meta list, containing the objects lengths and widths
    and the worker ip address, as the last value in the list.

    Parameters
    ----------
    num_returns : int
    args : tuple

    Returns
    -------
    Generator
    """
    chain = list(reversed(args))
    meta = []
    try:
        stack = [_construct_chain(chain, {}, meta, None)]
        while stack:
            try:
                gen = stack.pop()
                obj = next(gen)
                stack.append(gen)
                if isinstance(obj, GeneratorType):
                    stack.append(obj)
                else:
                    yield obj
            except StopIteration:
                ...
    except DeferredExecutionException as err:
        for _ in range(num_returns - 1):
            yield err
    except Exception as err:
        err = DeferredExecutionException(
            f"args={args}, chain={list(reversed(chain))}", err
        )
        for _ in range(num_returns - 1):
            yield err
    meta.append(get_node_ip_address())
    yield meta


def _construct_chain(
    chain: List,
    refs: Dict[int, Any],
    meta: List,
    lst: Optional[List],
):  # pragma: no cover # noqa: GL08
    pop = chain.pop
    tg_e = _Tag.END

    obj = pop()
    if isinstance(obj, DeferredExecutionException):
        raise obj
    if obj is _Tag.REF:
        obj = refs[pop()]
    elif obj is _Tag.LIST:
        obj = []
        yield _construct_list(obj, chain, refs, meta)
        if isinstance(obj[0], DeferredExecutionException):
            raise obj[0]

    while chain:
        fn = pop()
        if fn == tg_e:
            lst.append(obj)
            break

        if (args_len := pop()) >= 0:
            if args_len == 0:
                args = []
            else:
                args = chain[-args_len:]
                del chain[-args_len:]
                args.reverse()
        else:
            args = []
            yield _construct_list(args, chain, refs, meta)
        if (args_len := pop()) >= 0:
            kwargs = {pop(): pop() for _ in range(args_len)}
        else:
            values = []
            yield _construct_list(values, chain, refs, meta)
            kwargs = {pop(): v for v in values}

        obj = _exec_func(fn, obj, args, kwargs)

        if ref := pop():
            refs[ref] = obj
        if pop():
            if isinstance(obj, ListOrTuple):
                for o in obj:
                    meta.append(len(o) if hasattr(o, "__len__") else 0)
                    meta.append(len(o.columns) if hasattr(o, "columns") else 0)
                    yield o
            else:
                meta.append(len(obj) if hasattr(obj, "__len__") else 0)
                meta.append(len(obj.columns) if hasattr(obj, "columns") else 0)
                yield obj


def _construct_list(
    lst: List,
    chain: List,
    refs: Dict[int, Any],
    meta: List,
):  # pragma: no cover # noqa: GL08
    pop = chain.pop
    lst_append = lst.append
    while True:
        obj = pop()
        if isinstance(obj, _Tag):
            if obj == _Tag.END:
                break
            elif obj == _Tag.CHAIN:
                yield _construct_chain(chain, refs, meta, lst)
            elif obj == _Tag.LIST:
                lst_append([])
                yield _construct_list(lst[-1], chain, refs, meta)
            elif obj is _Tag.REF:
                lst_append(refs[pop()])
            else:
                raise ValueError(f"Unexpected tag {obj}")
        else:
            lst_append(obj)


@ray.remote(num_returns=4)
def remote_exec_func(
    fn: Callable, obj: Any, *args: Tuple, **kwargs: Dict
):  # pragma: no cover
    """
    Execute the specified function in a worker process.

    The object `obj` is passed to the function as the first argument of.

    Parameters
    ----------
    fn : Callable
    obj : Any
    *args : list
    **kwargs : dict

    Returns
    -------
    tuple[Any, int, int, str]
    The execution result, the result length and width, the worked address.
    """
    try:
        obj = _exec_func(fn, obj, args, kwargs)
        return (
            obj,
            len(obj) if hasattr(obj, "__len__") else 0,
            len(obj.columns) if hasattr(obj, "columns") else 0,
            get_node_ip_address(),
        )
    except DeferredExecutionException as err:
        return [err] * 4


def _exec_func(fn: Callable, obj: Any, args: Tuple, kwargs: Dict):  # noqa: GL08
    try:
        try:
            return fn(obj, *args, **kwargs)
            # Sometimes Arrow forces us to make a copy of an object before we operate on it. We
            # don't want the error to propagate to the user, and we want to avoid copying unless
            # we absolutely have to.
        except ValueError as err:
            if isinstance(obj, (pandas.DataFrame, pandas.Series)):
                return fn(obj.copy(), *args, **kwargs)
            else:
                raise err
    except Exception as err:
        raise DeferredExecutionException(
            f"fn={fn}, obj={obj}, args={args}, kwargs={kwargs}", err
        )


def _remote_exec_chain(num_returns: int, *args: Tuple):
    """
    Execute the deconstructed chain in a worker process.

    Parameters
    ----------
    num_returns : int
        The number of return values.
    *args : tuple
        A deconstructed chain to be executed.

    Returns
    -------
    Generator
    """
    if num_returns == 2:
        return _remote_exec_single_chain.remote(*args)
    else:
        return _remote_exec_multi_chain.options(num_returns=num_returns).remote(
            num_returns, *args
        )


@ray.remote(num_returns=2)
def _remote_exec_single_chain(*args: Tuple) -> Generator:  # pragma: no cover
    """
    Execute the deconstructed chain with a single return value in a worker process.

    Parameters
    ----------
    *args : tuple
        A deconstructed chain to be executed.

    Returns
    -------
    Generator
    """
    return construct(2, args)


@ray.remote
def _remote_exec_multi_chain(
    num_returns: int, *args: Tuple
) -> Generator:  # pragma: no cover
    """
    Execute the deconstructed chain with a multiple return values in a worker process.

    Parameters
    ----------
    num_returns : int
        The number of return values.
    *args : tuple
        A deconstructed chain to be executed.

    Returns
    -------
    Generator
    """
    return construct(num_returns, args)
