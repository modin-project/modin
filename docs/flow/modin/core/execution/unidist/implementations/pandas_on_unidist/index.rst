:orphan:

PandasOnUnidist Execution
=========================

Queries that perform data transformation, data ingress or data egress using the `pandas on Unidist` execution
pass through the Modin components detailed below.

To enable `pandas on Unidist` execution, please refer to the usage section in :doc:`pandas on Unidist </development/using_pandas_on_unidist>`.

Data Transformation
'''''''''''''''''''

.. image:: /img/pandas_on_unidist_data_transform.svg
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
the :py:class:`~modin.core.execution.unidist.implementations.pandas_on_unidist.dataframe.PandasOnUnidistDataframe` which inherits
generic functionality from the ``GenericUnidistDataframe`` and the :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe`.

..
  TODO: insert a link to ``GenericUnidistDataframe`` once we add an implementatiton of the class

PandasOnUnidist Dataframe implementation
----------------------------------------

Modin implements ``Dataframe``, ``PartitionManager``, ``VirtualPartition`` (a specific kind of ``AxisPartition`` with the capability
to combine smaller partitions into the one "virtual") and ``Partition`` classes specifically for the ``PandasOnUnidist`` execution:

* :doc:`PandasOnUnidistDataframe <dataframe>`
* :doc:`PandasOnUnidistDataframePartition <partitioning/partition>`
* :doc:`PandasOnUnidistDataframeVirtualPartition <partitioning/axis_partition>`
* :doc:`PandasOnUnidistDataframePartitionManager <partitioning/partition_manager>`

.. toctree::
    :hidden:

    dataframe
    partitioning/partition
    partitioning/axis_partition
    partitioning/partition_manager

Data Ingress
''''''''''''

.. image:: /img/pandas_on_unidist_data_ingress.svg
   :align: center

Data Egress
'''''''''''

.. image:: /img/pandas_on_unidist_data_egress.svg
   :align: center


When a user calls any IO function from the ``modin.pandas.io`` module, the `API` layer queries the
:py:class:`~modin.core.execution.dispatching.factories.dispatcher.FactoryDispatcher` which defines a factory specific for
the execution, namely, the :py:class:`~modin.core.execution.dispatching.factories.factories.PandasOnUnidistFactory`. The factory, in turn,
exposes the :py:class:`~modin.core.execution.unidist.implementations.pandas_on_unidist.io.PandasOnUnidistIO` class
whose responsibility is to perform a parallel read/write from/to a file.

When reading data from a CSV file, for example, the :py:class:`~modin.core.execution.unidist.implementations.pandas_on_unidist.io.PandasOnUnidistIO` class forwards
the user query to the :meth:`~modin.core.io.text.CSVDispatcher._read` method of :py:class:`~modin.core.io.text.CSVDispatcher`, where the query's parameters are preprocessed
to check if they are supported by the execution (defaulting to pandas if they are not) and computes some metadata
common for all partitions to be read. Then, the file is split into row chunks, and this data is used to launch remote tasks on the Unidist workers
via the :meth:`~modin.core.execution.unidist.common.UnidistWrapper.deploy` method of :py:class:`~modin.core.execution.unidist.common.UnidistWrapper`.
On each Unidist worker, the :py:class:`~modin.core.storage_formats.pandas.parsers.PandasCSVParser` parses data.
After the remote tasks are finished, additional result postprocessing is performed,
and a new query compiler with the data read is returned.

When writing data to a CSV file, for example, the :py:class:`~modin.core.execution.unidist.implementations.pandas_on_unidist.io.PandasOnUnidistIO` processes
the user query to execute it on Unidist workers. Then, the :py:class:`~modin.core.execution.unidist.implementations.pandas_on_unidist.io.PandasOnUnidistIO` asks the
:py:class:`~modin.core.execution.unidist.implementations.pandas_on_unidist.dataframe.PandasOnUnidistDataframe` to decompose the data into row-wise partitions
that will be written into the file in parallel in Unidist workers.