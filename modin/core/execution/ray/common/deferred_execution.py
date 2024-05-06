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

from enum import Enum
from itertools import islice
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

from modin.config import RayTaskCustomResources
from modin.core.execution.ray.common import MaterializationHook, RayWrapper
from modin.logging import get_logger

ObjectRefType = Union[ray.ObjectRef, None]
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
    the nodes that have multiple subscribers.

    Parameters
    ----------
    data : ObjectRefType or DeferredExecution
        The execution input.
    func : callable or ObjectRefType
        A function to be executed.
    args : list or tuple
        Additional positional arguments to be passed in `func`.
    kwargs : dict
        Additional keyword arguments to be passed in `func`.
    num_returns : int, optional
        The number of the return values.

    Attributes
    ----------
    data : ObjectRefType or DeferredExecution
        The execution input.
    func : callable or ObjectRefType
        A function to be executed.
    args : list or tuple
        Additional positional arguments to be passed in `func`.
    kwargs : dict
        Additional keyword arguments to be passed in `func`.
    num_returns : int
        The number of the return values.
    flat_args : bool
        True means that there are no lists or DeferredExecution objects in `args`.
        In this case, no arguments processing is performed and `args` is passed
        to the remote method as is.
    flat_kwargs : bool
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
        args: Union[List[Any], Tuple[Any]],
        kwargs: Dict[str, Any],
        num_returns=1,
    ):
        if isinstance(data, DeferredExecution):
            data.subscribe()
        self.data = data
        self.func = func
        self.args = args
        self.kwargs = kwargs
        self.num_returns = num_returns
        self.flat_args = self._flat_args(args)
        self.flat_kwargs = self._flat_args(kwargs.values())
        self.subscribers = 0

    @classmethod
    def _flat_args(cls, args: Iterable):
        """
        Check if the arguments list is flat and subscribe to all `DeferredExecution` objects.

        Parameters
        ----------
        args : Iterable

        Returns
        -------
        bool
        """
        flat = True
        for arg in args:
            if isinstance(arg, DeferredExecution):
                flat = False
                arg.subscribe()
            elif isinstance(arg, ListOrTuple):
                flat = False
                cls._flat_args(arg)
        return flat

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
        if self.has_result:
            return self.data, self.meta, self.meta_offset

        if (
            not isinstance(self.data, DeferredExecution)
            and self.flat_args
            and self.flat_kwargs
            and self.num_returns == 1
        ):
            result, length, width, ip = remote_exec_func.options(
                resources=RayTaskCustomResources.get()
            ).remote(self.func, self.data, *self.args, **self.kwargs)
            meta = MetaList([length, width, ip])
            self._set_result(result, meta, 0)
            return result, meta, 0

        # If there are no subscribers, we still need the result here. We don't need to decrement
        # it back. After the execution, the result is saved and the counter has no effect.
        self.subscribers += 2
        consumers, output = self._deconstruct()
        # The last result is the MetaList, so adding +1 here.
        num_returns = sum(c.num_returns for c in consumers) + 1
        results = self._remote_exec_chain(num_returns, *output)
        meta = MetaList(results.pop())
        meta_offset = 0
        results = iter(results)
        for de in consumers:
            if de.num_returns == 1:
                de._set_result(next(results), meta, meta_offset)
                meta_offset += 2
            else:
                res = list(islice(results, num_returns))
                offsets = list(range(0, 2 * num_returns, 2))
                de._set_result(res, meta, offsets)
                meta_offset += 2 * num_returns
        return self.data, self.meta, self.meta_offset

    @property
    def has_result(self):
        """
        Return true if this task has already been executed and the result is set.

        Returns
        -------
        bool
        """
        return not hasattr(self, "func")

    def subscribe(self):
        """
        Increment the `subscribers` counter.

        Subscriber is any instance that could trigger the execution of this task.
        In case of a multiple subscribers, the execution could be triggerred multiple
        times. To prevent the multiple executions, the execution result is returned
        from the worker and saved in this instance. Subsequent calls to `execute()`
        return the previously saved result.
        """
        self.subscribers += 1

    def unsubscribe(self):
        """Decrement the `subscribers` counter."""
        self.subscribers -= 1
        assert self.subscribers >= 0

    def _deconstruct(self) -> Tuple[List["DeferredExecution"], List[Any]]:
        """
        Convert the specified execution tree to a flat list.

        This is required for the automatic Ray object references
        materialization before passing the list to a Ray worker.

        The format of the list is the following:
        <input object> sequence<<function> <n><args> <n><kwargs> <ref> <nret>>...
        If <n> before <args> is >= 0, then the next n objects are the function arguments.
        If it is -1, it means that the method arguments contain list and/or
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

        <nret> field contains either the `num_returns` value or 0. If it's 0, the
        execution result is not returned, but is just passed to the next task in the
        chain. If it's 1, the result is returned as is. Otherwise, it's expected that
        the result is iterable and the specified number of values is returned from
        the iterator. The values lengths and widths are added to the meta list.

        Returns
        -------
        tuple of list
            * The first list is the result consumers.
                If a DeferredExecution has multiple subscribers, the execution result
                should be returned and saved in order to avoid duplicate executions.
                These DeferredExecution tasks are added to this list and, after the
                execution, the results are passed to the ``_set_result()`` method of
                each task.
            * The second is a flat list of arguments that could be passed to the remote executor.
        """
        stack = []
        result_consumers = []
        output = []
        # Using stack and generators to avoid the ``RecursionError``s.
        stack.append(self._deconstruct_chain(self, output, stack, result_consumers))
        while stack:
            try:
                gen = stack.pop()
                next_gen = next(gen)
                stack.append(gen)
                stack.append(next_gen)
            except StopIteration:
                pass
        return result_consumers, output

    @classmethod
    def _deconstruct_chain(
        cls,
        de: "DeferredExecution",
        output: List,
        stack: List,
        result_consumers: List["DeferredExecution"],
    ):
        """
        Deconstruct the specified DeferredExecution chain.

        Parameters
        ----------
        de : DeferredExecution
            The chain to be deconstructed.
        output : list
            Put the arguments to this list.
        stack : list
            Used to eliminate recursive calls, that may lead to the RecursionError.
        result_consumers : list of DeferredExecution
            The result consumers.

        Yields
        ------
        Generator
            The ``_deconstruct_list()`` generator.
        """
        out_append = output.append
        out_extend = output.extend
        while True:
            de.unsubscribe()
            if (out_pos := getattr(de, "out_pos", None)) and not de.has_result:
                out_append(_Tag.REF)
                out_append(out_pos)
                output[out_pos] = out_pos
                if de.subscribers == 0:
                    # We may have subscribed to the same node multiple times.
                    # It could happen, for example, if it's passed to the args
                    # multiple times, or it's one of the parent nodes and also
                    # passed to the args. In this case, there are no multiple
                    # subscribers, and we don't need to return the result.
                    output[out_pos + 1] = 0
                    result_consumers.remove(de)
                break
            elif not isinstance(data := de.data, DeferredExecution):
                if isinstance(data, ListOrTuple):
                    yield cls._deconstruct_list(
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

        while stack and isinstance(stack[-1], DeferredExecution):
            de: DeferredExecution = stack.pop()
            args = de.args
            kwargs = de.kwargs
            out_append(de.func)
            if de.flat_args:
                out_append(len(args))
                out_extend(args)
            else:
                out_append(-1)
                yield cls._deconstruct_list(
                    args, output, stack, result_consumers, out_append
                )
            if de.flat_kwargs:
                out_append(len(kwargs))
                for item in kwargs.items():
                    out_extend(item)
            else:
                out_append(-1)
                yield cls._deconstruct_list(
                    kwargs.values(), output, stack, result_consumers, out_append
                )
                out_extend(kwargs)

            out_append(0)  # Placeholder for ref id
            if de.subscribers > 0:
                # Ref id. This is the index in the output list.
                de.out_pos = len(output) - 1
                result_consumers.append(de)
                out_append(de.num_returns)  # Return result for this node
            else:
                out_append(0)  # Do not return result for this node

    @classmethod
    def _deconstruct_list(
        cls,
        lst: Iterable,
        output: List,
        stack: List,
        result_consumers: List["DeferredExecution"],
        out_append: Callable,
    ):
        """
        Deconstruct the specified list.

        Parameters
        ----------
        lst : list
        output : list
        stack : list
        result_consumers : list
        out_append : Callable
            The reference to the ``list.append()`` method.

        Yields
        ------
        Generator
            Either ``_deconstruct_list()`` or ``_deconstruct_chain()`` generator.
        """
        for obj in lst:
            if isinstance(obj, DeferredExecution):
                if out_pos := getattr(obj, "out_pos", None):
                    obj.unsubscribe()
                    if obj.has_result:
                        out_append(obj.data)
                    else:
                        out_append(_Tag.REF)
                        out_append(out_pos)
                        output[out_pos] = out_pos
                        if obj.subscribers == 0:
                            output[out_pos + 1] = 0
                            result_consumers.remove(obj)
                else:
                    out_append(_Tag.CHAIN)
                    yield cls._deconstruct_chain(obj, output, stack, result_consumers)
                    out_append(_Tag.END)
            elif isinstance(obj, ListOrTuple):
                out_append(_Tag.LIST)
                yield cls._deconstruct_list(
                    obj, output, stack, result_consumers, out_append
                )
            else:
                out_append(obj)
        out_append(_Tag.END)

    @staticmethod
    def _remote_exec_chain(num_returns: int, *args: Tuple) -> List[Any]:
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
        list
            The execution results. The last element of this list is the ``MetaList``.
        """
        # Prefer _remote_exec_single_chain(). It has fewer arguments and
        # does not require the num_returns to be specified in options.
        if num_returns == 2:
            return _remote_exec_single_chain.options(
                resources=RayTaskCustomResources.get()
            ).remote(*args)
        else:
            return _remote_exec_multi_chain.options(
                num_returns=num_returns, resources=RayTaskCustomResources.get()
            ).remote(num_returns, *args)

    def _set_result(
        self,
        result: ObjectRefOrListType,
        meta: "MetaList",
        meta_offset: Union[int, List[int]],
    ):
        """
        Set the execution result.

        Parameters
        ----------
        result : ObjectRefOrListType
        meta : MetaList
        meta_offset : int or list of int
        """
        del self.func, self.args, self.kwargs, self.flat_args, self.flat_kwargs
        self.data = result
        self.meta = meta
        self.meta_offset = meta_offset

    def __reduce__(self):
        """Not serializable."""
        raise NotImplementedError("DeferredExecution is not serializable!")


class MetaList:
    """
    Meta information, containing the result lengths and the worker address.

    Parameters
    ----------
    obj : ray.ObjectID or list
    """

    def __init__(self, obj: Union[ray.ObjectID, List]):
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
        return obj[index] if isinstance(obj, list) else MetaListHook(self, index)

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
            self._obj = obj = RayWrapper.materialize(obj)
        obj[index] = value


class MetaListHook(MaterializationHook):
    """
    Used by MetaList.__getitem__() for lazy materialization and getting a single value from the list.

    Parameters
    ----------
    meta : MetaList
        Non-materialized list to get the value from.
    idx : int
        The value index in the list.
    """

    def __init__(self, meta: MetaList, idx: int):
        self.meta = meta
        self.idx = idx

    def pre_materialize(self):
        """
        Get item at self.idx or object ref if not materialized.

        Returns
        -------
        object
        """
        obj = self.meta._obj
        return obj[self.idx] if isinstance(obj, list) else obj

    def post_materialize(self, materialized):
        """
        Save the materialized list in self.meta and get the item at self.idx.

        Parameters
        ----------
        materialized : list

        Returns
        -------
        object
        """
        self.meta._obj = materialized
        return materialized[self.idx]


class _Tag(Enum):  # noqa: PR01
    """
    A set of special values used for the method arguments de/construction.

    See ``DeferredExecution._deconstruct()`` for details.
    """

    # The next item is an execution chain
    CHAIN = 0
    # The next item is a reference
    REF = 1
    # The next item a list
    LIST = 2
    # End of list or chain
    END = 3


class _RemoteExecutor:
    """Remote functions for DeferredExecution."""

    @staticmethod
    def exec_func(fn: Callable, obj: Any, args: Tuple, kwargs: Dict) -> Any:
        """
        Execute the specified function.

        Parameters
        ----------
        fn : Callable
        obj : Any
        args : Tuple
        kwargs : dict

        Returns
        -------
        Any
        """
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
            get_logger().error(
                f"{err}. fn={fn}, obj={obj}, args={args}, kwargs={kwargs}"
            )
            raise err

    @classmethod
    def construct(cls, num_returns: int, args: Tuple):  # pragma: no cover
        """
        Construct and execute the specified chain.

        This function is called in a worker process. The last value, returned by
        this generator, is the meta list, containing the objects lengths and widths
        and the worker ip address, as the last value in the list.

        Parameters
        ----------
        num_returns : int
        args : tuple

        Yields
        ------
        Any
            The execution results and the MetaList as the last value.
        """
        chain = list(reversed(args))
        meta = []
        try:
            stack = [cls.construct_chain(chain, {}, meta, None)]
            while stack:
                try:
                    gen = stack.pop()
                    obj = next(gen)
                    stack.append(gen)
                    if isinstance(obj, Generator):
                        stack.append(obj)
                    else:
                        yield obj
                except StopIteration:
                    pass
        except Exception as err:
            get_logger().error(f"{err}. args={args}, chain={list(reversed(chain))}")
            raise err
        meta.append(get_node_ip_address())
        yield meta

    @classmethod
    def construct_chain(
        cls,
        chain: List,
        refs: Dict[int, Any],
        meta: List,
        lst: Optional[List],
    ):  # pragma: no cover
        """
        Construct the chain and execute it one by one.

        Parameters
        ----------
        chain : list
            A flat list containing the execution tree, deconstructed by
            ``DeferredExecution._deconstruct()``.
        refs : dict
            If an execution result is required for multiple chains, the
            reference to this result is saved in this dict.
        meta : list
            The lengths of the returned objects are added to this list.
        lst : list
            If specified, the execution result is added to this list.
            This is used when a chain is passed as an argument to a
            DeferredExecution task.

        Yields
        ------
        Any
            Either the ``construct_list()`` generator or the execution results.
        """
        pop = chain.pop
        tg_e = _Tag.END

        obj = pop()
        if obj is _Tag.REF:
            obj = refs[pop()]
        elif obj is _Tag.LIST:
            obj = []
            yield cls.construct_list(obj, chain, refs, meta)

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
                yield cls.construct_list(args, chain, refs, meta)
            if (args_len := pop()) >= 0:
                kwargs = {pop(): pop() for _ in range(args_len)}
            else:
                values = []
                yield cls.construct_list(values, chain, refs, meta)
                kwargs = {pop(): v for v in values}

            obj = cls.exec_func(fn, obj, args, kwargs)

            if ref := pop():  # <ref> is not 0 - adding the result to refs
                refs[ref] = obj
            if (num_returns := pop()) == 0:
                continue

            itr = iter([obj] if num_returns == 1 else obj)
            for _ in range(num_returns):
                obj = next(itr)
                meta.append(len(obj) if hasattr(obj, "__len__") else 0)
                meta.append(len(obj.columns) if hasattr(obj, "columns") else 0)
                yield obj

    @classmethod
    def construct_list(
        cls,
        lst: List,
        chain: List,
        refs: Dict[int, Any],
        meta: List,
    ):  # pragma: no cover
        """
        Construct the list.

        Parameters
        ----------
        lst : list
        chain : list
        refs : dict
        meta : list

        Yields
        ------
        Any
            Either ``construct_chain()`` or ``construct_list()`` generator.
        """
        pop = chain.pop
        lst_append = lst.append
        while True:
            obj = pop()
            if isinstance(obj, _Tag):
                if obj == _Tag.END:
                    break
                elif obj == _Tag.CHAIN:
                    yield cls.construct_chain(chain, refs, meta, lst)
                elif obj == _Tag.LIST:
                    lst_append([])
                    yield cls.construct_list(lst[-1], chain, refs, meta)
                elif obj is _Tag.REF:
                    lst_append(refs[pop()])
                else:
                    raise ValueError(f"Unexpected tag {obj}")
            else:
                lst_append(obj)

    def __reduce__(self):
        """
        Use a single instance on deserialization.

        Returns
        -------
        str
            Returns the ``_REMOTE_EXEC`` attribute name.
        """
        return "_REMOTE_EXEC"


_REMOTE_EXEC = _RemoteExecutor()


@ray.remote(num_returns=4)
def remote_exec_func(
    fn: Callable,
    obj: Any,
    *flat_args: Tuple,
    remote_executor=_REMOTE_EXEC,
    **flat_kwargs: Dict,
):  # pragma: no cover
    """
    Execute the specified function with the arguments in a worker process.

    The object `obj` is passed to the function as the first argument.
    Note: all the arguments must be flat, i.e. no lists, no chains.

    Parameters
    ----------
    fn : Callable
    obj : Any
    *flat_args : list
    remote_executor : _RemoteExecutor, default: _REMOTE_EXEC
        Do not change, it's used to avoid excessive serializations.
    **flat_kwargs : dict

    Returns
    -------
    tuple[Any, int, int, str]
    The execution result, the result length and width, the worked address.
    """
    obj = remote_executor.exec_func(fn, obj, flat_args, flat_kwargs)
    return (
        obj,
        len(obj) if hasattr(obj, "__len__") else 0,
        len(obj.columns) if hasattr(obj, "columns") else 0,
        get_node_ip_address(),
    )


@ray.remote(num_returns=2)
def _remote_exec_single_chain(
    *args: Tuple, remote_executor=_REMOTE_EXEC
) -> Generator:  # pragma: no cover
    """
    Execute the deconstructed chain with a single return value in a worker process.

    Parameters
    ----------
    *args : tuple
        A deconstructed chain to be executed.
    remote_executor : _RemoteExecutor, default: _REMOTE_EXEC
        Do not change, it's used to avoid excessive serializations.

    Returns
    -------
    Generator
    """
    return remote_executor.construct(num_returns=2, args=args)


@ray.remote
def _remote_exec_multi_chain(
    num_returns: int, *args: Tuple, remote_executor=_REMOTE_EXEC
) -> Generator:  # pragma: no cover
    """
    Execute the deconstructed chain with a multiple return values in a worker process.

    Parameters
    ----------
    num_returns : int
        The number of return values.
    *args : tuple
        A deconstructed chain to be executed.
    remote_executor : _RemoteExecutor, default: _REMOTE_EXEC
        Do not change, it's used to avoid excessive serializations.

    Returns
    -------
    Generator
    """
    return remote_executor.construct(num_returns, args)
