:orphan:

PandasOnPython Execution
========================

Queries that perform data transformation, data ingress or data egress using the `pandas on Python` execution
pass through the Modin components detailed below.

`pandas on Python` execution is sequential and it's used for the debug purposes. To enable `pandas on Python` execution,
please refer to the usage section in :doc:`pandas on Python </development/using_pandas_on_python>`.

Data Transformation
'''''''''''''''''''

.. image:: /img/pandas_on_python_data_transform.svg
   :align: center

When a user calls any :py:class:`~modin.pandas.dataframe.DataFrame` API, a query starts forming at the `API` layer
to be executed at the `Execution` layer. The `API` layer is responsible for processing the query appropriately,
for example, determining whether the final result should be a ``DataFrame`` or ``Series`` object. This layer is also responsible for sanitizing the input to the
:py:class:`~modin.core.storage_formats.pandas.query_compiler.PandasQueryCompiler`, e.g. validating a parameter from the query
and defining specific intermediate values to provide more context to the query compiler.
The :py:class:`~modin.core.storage_formats.pandas.query_compiler.PandasQueryCompiler` is responsible for
processing the query, received from the :py:class:`~modin.pandas.dataframe.DataFrame` `API` layer,
to determine how to apply it to a subset of the data - either cell-wise or along an axis-wise partition backed by the `pandas`
storage format. The :py:class:`~modin.core.storage_formats.pandas.query_compiler.PandasQueryCompiler` maps the query to one of the :doc:`Core Algebra Operators </flow/modin/core/dataframe/algebra>` of
the :py:class:`~modin.core.execution.python.implementations.pandas_on_python.dataframe.dataframe.PandasOnPythonDataframe` which inherits
generic functionality from the :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe`.

PandasOnPython Dataframe implementation
---------------------------------------

This page describes implementation of :doc:`Modin PandasDataframe Objects </flow/modin/core/dataframe/pandas/index>`
specific for `PandasOnPython` execution. Since Python engine doesn't allow computation parallelization,
operations on partitions are performed sequentially. The absence of parallelization doesn't give any
performance speed-up, so ``PandasOnPython`` is used for testing purposes only.

* :doc:`PandasOnPythonDataframe <dataframe>`
* :doc:`PandasOnPythonDataframePartition <partitioning/partition>`
* :doc:`PandasOnPythonDataframeAxisPartition <partitioning/axis_partition>`
* :doc:`PandasOnPythonDataframePartitionManager <partitioning/partition_manager>`

.. toctree::
    :hidden:

    dataframe
    partitioning/partition
    partitioning/axis_partition
    partitioning/partition_manager


Data Ingress
''''''''''''

.. image:: /img/pandas_on_python_data_ingress.svg
   :align: center

Data Egress
'''''''''''

.. image:: /img/pandas_on_python_data_egress.svg
   :align: center


When a user calls any IO function from the ``modin.pandas.io`` module, the `API` layer queries the
:py:class:`~modin.core.execution.dispatching.factories.dispatcher.FactoryDispatcher` which defines a factory specific for
the execution, namely, the :py:class:`~modin.core.execution.dispatching.factories.factories.PandasOnPythonFactory`. The factory, in turn,
exposes the :py:class:`~modin.core.execution.python.implementations.pandas_on_python.io.PandasOnPythonIO` class
whose responsibility is a read/write from/to a file.

When reading data from a CSV file, for example, the :py:class:`~modin.core.execution.python.implementations.pandas_on_python.io.io.PandasOnPythonIO` class
reads the data using corresponding `pandas` function (``pandas.read_csv()`` in this case). After the reading is complete, a new query compiler is created from `pandas` object
using :py:meth:`~modin.core.execution.python.implementations.pandas_on_python.io.io.PandasOnPythonIO.from_pandas` and returned.

When writing data to a CSV file, for example, the :py:class:`~modin.core.execution.python.implementations.pandas_on_python.io.PandasOnPythonIO` converts a query compiler
to `pandas` object using :py:meth:`~modin.core.storage_formats.base.query_compiler.BaseQueryCompiler.to_pandas`. After that, `pandas` writes the data to the file using
corresponding function (``pandas.to_csv()`` in this case).