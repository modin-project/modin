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

from modin.backends.pandas.query_compiler import PandasQueryCompiler
import pandas
from pandas.core.computation.expr import Expr
from pandas.core.computation.scope import Scope
from pandas.core.computation.ops import UnaryOp, BinOp, Term, MathCall, Constant

import pyarrow as pa
import pyarrow.gandiva as gandiva


class FakeSeries:
    def __init__(self, dtype):
        self.dtype = dtype


class PyarrowQueryCompiler(PandasQueryCompiler):
    def query(self, expr, **kwargs):
        """Query columns of the QueryCompiler with a boolean expression.

        Args:
            expr: Boolean expression to query the columns with.

        Returns:
            QueryCompiler containing the rows where the boolean expression is satisfied.
        """

        def gen_table_expr(table, expr):
            resolver = {
                name: FakeSeries(dtype.to_pandas_dtype())
                for name, dtype in zip(table.schema.names, table.schema.types)
            }
            scope = Scope(level=0, resolvers=(resolver,))
            return Expr(expr=expr, env=scope)

        unary_ops = {"~": "not"}
        math_calls = {"log": "log", "exp": "exp", "log10": "log10", "cbrt": "cbrt"}
        bin_ops = {
            "+": "add",
            "-": "subtract",
            "*": "multiply",
            "/": "divide",
            "**": "power",
        }
        cmp_ops = {
            "==": "equal",
            "!=": "not_equal",
            ">": "greater_than",
            "<": "less_than",
            "<=": "less_than_or_equal_to",
            ">": "greater_than",
            ">=": "greater_than_or_equal_to",
            "like": "like",
        }

        def build_node(table, terms, builder):
            if isinstance(terms, Constant):
                return builder.make_literal(
                    terms.value, (pa.from_numpy_dtype(terms.return_type))
                )

            if isinstance(terms, Term):
                return builder.make_field(table.schema.field_by_name(terms.name))

            if isinstance(terms, BinOp):
                lnode = build_node(table, terms.lhs, builder)
                rnode = build_node(table, terms.rhs, builder)
                return_type = pa.from_numpy_dtype(terms.return_type)

                if terms.op == "&":
                    return builder.make_and([lnode, rnode])
                if terms.op == "|":
                    return builder.make_or([lnode, rnode])
                if terms.op in cmp_ops:
                    assert return_type == pa.bool_()
                    return builder.make_function(
                        cmp_ops[terms.op], [lnode, rnode], return_type
                    )
                if terms.op in bin_ops:
                    return builder.make_function(
                        bin_ops[terms.op], [lnode, rnode], return_type
                    )

            if isinstance(terms, UnaryOp):
                return_type = pa.from_numpy_dtype(terms.return_type)
                return builder.make_function(
                    unary_ops[terms.op],
                    [build_node(table, terms.operand, builder)],
                    return_type,
                )

            if isinstance(terms, MathCall):
                return_type = pa.from_numpy_dtype(terms.return_type)
                childern = [
                    build_node(table, child, builder) for child in terms.operands
                ]
                return builder.make_function(
                    math_calls[terms.op], childern, return_type
                )

            raise TypeError("Unsupported term type: %s" % terms)

        def can_be_condition(expr):
            if isinstance(expr.terms, BinOp):
                if expr.terms.op in cmp_ops or expr.terms.op in ("&", "|"):
                    return True
            elif isinstance(expr.terms, UnaryOp):
                if expr.terms.op == "~":
                    return True
            return False

        def filter_with_selection_vector(table, s):
            record_batch = table.to_batches()[0]
            indices = s.to_array()  # .to_numpy()
            new_columns = [
                pa.array(c.to_numpy()[indices]) for c in record_batch.columns
            ]
            return pa.Table.from_arrays(new_columns, record_batch.schema.names)

        def gandiva_query(table, query):
            expr = gen_table_expr(table, query)
            if not can_be_condition(expr):
                raise ValueError("Root operation should be a filter.")
            builder = gandiva.TreeExprBuilder()
            root = build_node(table, expr.terms, builder)
            cond = builder.make_condition(root)
            filt = gandiva.make_filter(table.schema, cond)
            sel_vec = filt.evaluate(table.to_batches()[0], pa.default_memory_pool())
            result = filter_with_selection_vector(table, sel_vec)
            return result

        def gandiva_query2(table, query):
            expr = gen_table_expr(table, query)
            if not can_be_condition(expr):
                raise ValueError("Root operation should be a filter.")
            builder = gandiva.TreeExprBuilder()
            root = build_node(table, expr.terms, builder)
            cond = builder.make_condition(root)
            filt = gandiva.make_filter(table.schema, cond)
            return filt

        def query_builder(arrow_table, **kwargs):
            return gandiva_query(arrow_table, kwargs.get("expr", ""))

        kwargs["expr"] = expr
        func = self._prepare_method(query_builder, **kwargs)
        new_data = self._map_across_full_axis(1, func)
        # Query removes rows, so we need to update the index
        new_index = self._compute_index(0, new_data, False)
        return self.__constructor__(
            new_data, new_index, self.columns, self._dtype_cache
        )

    def _compute_index(self, axis, data_object, compute_diff=True):
        def arrow_index_extraction(table, axis):
            if not axis:
                return pandas.Index(table.column(table.num_columns - 1))
            else:
                try:
                    return pandas.Index(table.columns)
                except AttributeError:
                    return []

        index_obj = self.index if not axis else self.columns
        old_blocks = self.data if compute_diff else None
        new_indices = data_object.get_indices(
            axis=axis,
            index_func=lambda df: arrow_index_extraction(df, axis),
            old_blocks=old_blocks,
        )
        return index_obj[new_indices] if compute_diff else new_indices

    def to_pandas(self):
        """Converts Modin DataFrame to Pandas DataFrame.

        Returns:
            Pandas DataFrame of the QueryCompiler.
        """
        return self._modin_frame.to_pandas()

    def to_numpy(self, **kwargs):
        """
        Converts Modin DataFrame to NumPy array.

        Returns
        -------
            NumPy array of the QueryCompiler.
        """
        return self._modin_frame.to_numpy(**kwargs)
