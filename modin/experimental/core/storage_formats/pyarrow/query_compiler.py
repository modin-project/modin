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
Module contains ``PyarrowQueryCompiler`` class.

``PyarrowQueryCompiler`` is responsible for compiling efficient DataFrame algebra
queries for the ``PyarrowOnRayDataframe``.
"""

import pandas
import pyarrow as pa

from modin.core.storage_formats.pandas.query_compiler import PandasQueryCompiler
from modin.utils import _inherit_docstrings
from pandas.core.computation.expr import Expr
from pandas.core.computation.scope import Scope
from pandas.core.computation.ops import UnaryOp, BinOp, Term, MathCall, Constant


class FakeSeries:
    """
    Series metadata class.

    Parameters
    ----------
    dtype : dtype
        Data-type of the represented Series.
    """

    def __init__(self, dtype):
        self.dtype = dtype


@_inherit_docstrings(PandasQueryCompiler)
class PyarrowQueryCompiler(PandasQueryCompiler):
    """
    Query compiler for the PyArrow storage format.

    This class translates common query compiler API into the DataFrame Algebra
    queries, that is supposed to be executed by
    :py:class:`~modin.experimental.core.execution.ray.implementations.pyarrow_on_ray.dataframe.dataframe.PyarrowOnRayDataframe`.

    Parameters
    ----------
    modin_frame : PyarrowOnRayDataframe
        Modin Frame to query with the compiled queries.
    """

    def query(self, expr, **kwargs):
        def gen_table_expr(table, expr):
            """
            Build pandas expression for the specified query.

            Parameters
            ----------
            table : pyarrow.Table
                Table to evaluate expression on.
            expr : str
                Query string to evaluate on the `table` columns.

            Returns
            -------
            pandas.core.computation.expr.Expr
            """
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
            "<": "less_than",
            "<=": "less_than_or_equal_to",
            ">": "greater_than",
            ">=": "greater_than_or_equal_to",
            "like": "like",
        }

        def build_node(table, terms, builder):
            """
            Build expression Node in Gandiva notation for the specified pandas expression.

            Parameters
            ----------
            table : pyarrow.Table
                Table to evaluate expression on.
            terms : pandas.core.computation.expr.Term
                Pandas expression to evaluate.
            builder : pyarrow.gandiva.TreeExprBuilder
                Pyarrow node builder.

            Returns
            -------
            pyarrow.gandiva.Node
            """
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
            """
            Check whether the passed expression is a conditional operation.

            Parameters
            ----------
            expr : pandas.core.computation.expr.Expr

            Returns
            -------
            bool
            """
            if isinstance(expr.terms, BinOp):
                if expr.terms.op in cmp_ops or expr.terms.op in ("&", "|"):
                    return True
            elif isinstance(expr.terms, UnaryOp):
                if expr.terms.op == "~":
                    return True
            return False

        def filter_with_selection_vector(table, s):
            """
            Filter passed pyarrow table with the specified filter.

            Parameters
            ----------
            table : pyarrow.Table
            s : pyarrow.gandiva.SelectionVector

            Returns
            -------
            pyarrow.Table
            """
            record_batch = table.to_batches()[0]
            indices = s.to_array()  # .to_numpy()
            new_columns = [
                pa.array(c.to_numpy()[indices]) for c in record_batch.columns
            ]
            return pa.Table.from_arrays(new_columns, record_batch.schema.names)

        def gandiva_query(table, query):
            """
            Evaluate string query on the passed table.

            Parameters
            ----------
            table : pyarrow.Table
                Table to evaluate query on.
            query : str
                Query string to evaluate on the `table` columns.

            Returns
            -------
            pyarrow.Table
            """
            expr = gen_table_expr(table, query)
            if not can_be_condition(expr):
                raise ValueError("Root operation should be a filter.")

            # We use this import here because of https://github.com/modin-project/modin/issues/3849,
            # after the issue is fixed we should put the import at the top of this file
            import pyarrow.gandiva as gandiva

            builder = gandiva.TreeExprBuilder()
            root = build_node(table, expr.terms, builder)
            cond = builder.make_condition(root)
            filt = gandiva.make_filter(table.schema, cond)
            sel_vec = filt.evaluate(table.to_batches()[0], pa.default_memory_pool())
            result = filter_with_selection_vector(table, sel_vec)
            return result

        def query_builder(arrow_table, **kwargs):
            """Evaluate string query on the passed pyarrow table."""
            return gandiva_query(arrow_table, kwargs.get("expr", ""))

        kwargs["expr"] = expr
        # FIXME: `PandasQueryCompiler._prepare_method` was removed in #721,
        # it is no longer needed to wrap function to apply.
        func = self._prepare_method(query_builder, **kwargs)
        # FIXME: `PandasQueryCompiler._map_across_full_axis` was removed in #721.
        # This method call should be replaced to its equivalent from `operators.function`
        new_data = self._map_across_full_axis(1, func)
        # Query removes rows, so we need to update the index
        new_index = self._compute_index(0, new_data, False)
        return self.__constructor__(
            new_data, new_index, self.columns, self._dtype_cache
        )

    def _compute_index(self, axis, data_object, compute_diff=True):
        """
        Compute index labels of the passed Modin Frame along specified axis.

        Parameters
        ----------
        axis : {0, 1}
            Axis to compute index labels along. 0 is for index and 1 is for column.
        data_object : PyarrowOnRayDataframe
            Modin Frame object to build indices from.
        compute_diff : bool, default: True
            Whether to cut the resulted indices to a subset of the self indices.

        Returns
        -------
        pandas.Index
        """

        def arrow_index_extraction(table, axis):
            """Extract index labels from the passed pyarrow table the along specified axis."""
            if not axis:
                return pandas.Index(table.column(table.num_columns - 1))
            else:
                try:
                    return pandas.Index(table.columns)
                except AttributeError:
                    return []

        index_obj = self.index if not axis else self.columns
        old_blocks = self.data if compute_diff else None
        # FIXME: `PandasDataframe.get_indices` was deprecated, this call should be
        # replaced either by `PandasDataframe._compute_axis_label` or by `PandasDataframe.axes`.
        new_indices = data_object.get_indices(
            axis=axis,
            index_func=lambda df: arrow_index_extraction(df, axis),
            old_blocks=old_blocks,
        )
        return index_obj[new_indices] if compute_diff else new_indices
