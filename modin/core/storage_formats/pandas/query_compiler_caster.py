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

"""
Module contains ``QueryCompilerCaster`` class.

``QueryCompilerCaster`` is used for automatically casting query compiler
arguments to the type of the current query compiler for query compiler class functions.
This ensures compatibility between different query compiler classes.
"""

import functools
import inspect
from itertools import combinations
from types import FunctionType, MethodType
from typing import Any, Dict, Tuple, TypeVar

from pandas.core.indexes.frozen import FrozenList

from modin.core.storage_formats.base.query_compiler import BaseQueryCompiler, QCCoercionCost

Fn = TypeVar("Fn", bound=Any)

class QueryCompilerCasterCalculator:
    
    def __init__(self):
        self._caster_costing_map = {}
        self._data_cls_map = {}
        self._qc_list = []
        self._qc_cls_list = []
        self._result_type = None
    
    def add_query_compiler(self, query_compiler):
        if isinstance(query_compiler, type):
            # class
            qc_type = query_compiler
        else:
            # instance
            qc_type = type(query_compiler)
            self._qc_list.append(query_compiler)
            self._data_cls_map[qc_type] = query_compiler._modin_frame
        self._qc_cls_list.append(qc_type)
    
    def calculate(self):
        if self._result_type is not None:
            return self._result_type
        if len(self._qc_cls_list) == 1:
            return self._qc_cls_list[0]
        if len(self._qc_cls_list) == 0:
            raise ValueError("No query compilers registered")
        
        for (qc_1, qc_2) in combinations(self._qc_list, 2):
            costs_1 = qc_1.qc_engine_switch_cost(qc_2)
            costs_2 = qc_2.qc_engine_switch_cost(qc_1)
            self._add_cost_data(costs_1)
            self._add_cost_data(costs_2)
        if len(self._caster_costing_map) <= 0 and len(self._qc_cls_list) > 0:
            self._result_type = self._qc_cls_list[0]
            return self._result_type
        min_value = min(self._caster_costing_map.values())
        for key, value in self._caster_costing_map.items():
            if min_value == value:
                self._result_type = key
                break
        return self._result_type
            
    def _add_cost_data(self, costs:dict):
        for k, v in costs.items():
            # filter out any extranious query compilers not in this operation
            if k in self._qc_cls_list:
                QCCoercionCost.validate_coercsion_cost(v)
                # Adds the costs associated with all coercions to a type, k
                self._caster_costing_map[k] = v + self._caster_costing_map[k] if k in self._caster_costing_map else v
    
    def result_data_frame(self):
        qc_type = self.calculate()
        return self._data_cls_map[qc_type]
        
    
class QueryCompilerCaster:
    """Cast all query compiler arguments of the member function to current query compiler."""

    @classmethod
    def __init_subclass__(
        cls,
        **kwargs: Dict,
    ) -> None:
        """
        Apply type casting to all children of ``QueryCompilerCaster``.

        This method is called automatically when a class inherits from
        ``QueryCompilerCaster``. It ensures that all member functions within the
        subclass have their arguments automatically casted to the current query
        compiler type.

        Parameters
        ----------
        **kwargs : Additional keyword arguments
        """
        super().__init_subclass__(**kwargs)
        apply_argument_cast(cls)


def visit_nested_args(arguments, 
                      current_qc:BaseQueryCompiler, 
                      fn:callable):
    """
    Cast all arguments in nested fashion to current query compiler.

    Parameters
    ----------
    arguments : tuple or dict
    current_qc : BaseQueryCompiler

    Returns
    -------
    tuple or dict
        Returns args and kwargs with all query compilers casted to current_qc.
    """

    imutable_types = (FrozenList, tuple)
    if isinstance(arguments, imutable_types):
        args_type = type(arguments)
        arguments = list(arguments)
        arguments = visit_nested_args(arguments, current_qc, fn)

        return args_type(arguments)
    if isinstance(arguments, list):
        for i in range(len(arguments)):
            if isinstance(arguments[i], (list, dict)):
                visit_nested_args(arguments[i], current_qc, fn)
            else:
                arguments[i] = fn(arguments[i])
    elif isinstance(arguments, dict):
        for key in arguments:
            if isinstance(arguments[key], (list, dict)):
                visit_nested_args(arguments[key], current_qc, fn)
            else:
                arguments[key] = fn(arguments[key])
    return arguments


def apply_argument_cast(obj: Fn) -> Fn:
    """
    Cast all arguments that are query compilers to the current query compiler.

    Parameters
    ----------
    obj : function

    Returns
    -------
    function
        Returns decorated function which does argument casting.
    """
    if isinstance(obj, type):
        all_attrs = dict(inspect.getmembers(obj))


        # This is required because inspect converts class methods to member functions
        current_class_attrs = vars(obj)
        for key in current_class_attrs:
            all_attrs[key] = current_class_attrs[key]
        all_attrs.pop("__abstractmethods__")
        all_attrs.pop("__init__")
        all_attrs.pop("qc_engine_switch_cost")
        all_attrs.pop("from_pandas")
        for attr_name, attr_value in all_attrs.items():
            if isinstance(
                attr_value, (FunctionType, MethodType, classmethod, staticmethod)
            ):
                wrapped = apply_argument_cast(attr_value)
                setattr(obj, attr_name, wrapped)
        return obj  # type: ignore [return-value]
    elif isinstance(obj, classmethod):
        return classmethod(apply_argument_cast(obj.__func__))  # type: ignore [return-value, arg-type]
    elif isinstance(obj, staticmethod):
        return staticmethod(apply_argument_cast(obj.__func__))

    @functools.wraps(obj)
    def cast_args(*args: Tuple, **kwargs: Dict) -> Any:
        """
        Add casting for query compiler arguments.

        Parameters
        ----------
        *args : tuple
            The function arguments.
        **kwargs : dict
            The function keyword arguments.

        Returns
        -------
        Any
        """
        if len(args) == 0 and len(kwargs) == 0:
            return
        current_qc = args[0]
        calculator = QueryCompilerCasterCalculator()
        calculator.add_query_compiler(current_qc)

        def arg_needs_casting(arg):
            current_qc_type = type(current_qc)
            if not isinstance(arg, BaseQueryCompiler):
                return False
            if isinstance(arg, current_qc_type):
                return False
            return True
        
        def register_query_compilers(arg):
            if not arg_needs_casting(arg):
                return arg
            calculator.add_query_compiler(arg)
            return arg
        
        def cast_to_qc(arg):
            if not arg_needs_casting(arg):
                return arg
            qc_type = calculator.calculate()
            if qc_type == None or qc_type == type(arg):
                return arg
            frame_data = calculator.result_data_frame()
            result = qc_type.from_pandas(arg.to_pandas(), frame_data)
            return result
        
            
        if isinstance(current_qc, BaseQueryCompiler):
            visit_nested_args(kwargs, current_qc, register_query_compilers)
            visit_nested_args(args, current_qc, register_query_compilers)
            
            args = visit_nested_args(args, current_qc, cast_to_qc)
            kwargs = visit_nested_args(kwargs, current_qc, cast_to_qc)

        
        qc = calculator.calculate()

        if qc == None or qc == type(current_qc):
            return obj(*args, **kwargs)

        #breakpoint()
        # we need to cast current_qc to a new query compiler
        if qc != current_qc:
            data_cls = current_qc._modin_frame
            return qc.from_pandas(current_qc.to_pandas(), data_cls)
        # need to find the new function for obj
        return obj(*args, **kwargs)

    return cast_args
