PandasOnRay Execution
=====================

Flows which a user’s queries for data transformation, data ingress and egress pass through the Modin components
for pandas on Ray execution are shown below.

To enable the execution, please refer to :doc:`pandas on Ray </UsingPandasonRay/index>` usage section.

Data Transformation
'''''''''''''''''''

.. image:: /img/pandas_on_ray_data_transform.svg
   :align: center

When a user calls any of :py:class:`~modin.pandas.dataframe.DataFrame` API, a query starts forming at the `API` layer
to be executed at the `Execution` layer. The `API` layer’s responsibility is to process the query in an appropriate way
for the final result (like, detect whether ``DataFrame`` or ``Series`` should be returned) and to ensure clean input to
:py:class:`~modin.core.storage_formats.pandas.query_compiler.PandasQueryCompiler` (like, validate a parameter of the call
to define specific values for it to pass that information to the query compiler).
:py:class:`~modin.core.storage_formats.pandas.query_compiler.PandasQueryCompiler` is in charge of
processing passed query in a way that is conforming for the subset of data (cell-wise or axis-wise partition) of `pandas`
storage format and mapping the query to one of the :doc:`Core Operators </flow/modin/core/dataframe/algebra>` of
:py:class:`~modin.core.execution.ray.implementations.pandas_on_ray.dataframe.dataframe.PandasOnRayDataframe` that inherits
generic functionality from ``GenericRayDataframe`` and :py:class:`~modin.core.dataframe.pandas.dataframe.dataframe.PandasDataframe`.

PandasOnRay Dataframe implementation
------------------------------------

Modin implements ``Dataframe``, ``PartitionManager``, ``AxisPartition`` and ``Partition`` classes
specific for ``PandasOnRay`` execution:

* :doc:`PandasOnRayDataframe <dataframe>`
* :doc:`PandasOnRayDataframePartition <partitioning/partition>`
* :doc:`PandasOnRayDataframeAxisPartition <partitioning/axis_partition>`
* :doc:`PandasOnRayDataframePartitionManager <partitioning/partition_manager>`

.. toctree::
    :hidden:

    dataframe
    partitioning/partition
    partitioning/axis_partition
    partitioning/partition_manager

Data Ingress
''''''''''''

.. image:: /img/pandas_on_ray_data_ingress.svg
   :align: center

Data Egress
'''''''''''

.. image:: /img/pandas_on_ray_data_egress.svg
   :align: center


When a user calls any of IO functions from ``modin.pandas.io`` module, `API` layer appeals to
:py:class:`~modin.core.execution.dispatching.factories.dispatcher.FactoryDispatcher` that defines a factory specific for
the execution, namely, :py:class:`~modin.core.execution.dispatching.factories.factories.PandasOnRayFactory`. The factory, in turn,
holds :py:class:`~modin.core.execution.ray.implementations.pandas_on_ray.io.PandasOnRayIO` class whose responsibility is parallel read/write
from/to a file.

In the event of reading data from a CSV file, for example, :py:class:`~modin.core.execution.ray.implementations.pandas_on_ray.io.io.PandasOnRayIO` class forwards
the user query to the :meth:`~modin.core.io.text.CSVDispatcher._read` method of :py:class:`~modin.core.io.text.CSVDispatcher`, where the query's parameters are preprocessed
to check if they are supported by the execution (otherwise, default pandas implementation is used to read the CSV file) and compute some metadata
common for all partitions to be read. Then, the file is split into row chunks and using this data remote tasks are launched on the Ray workers
via :meth:`~modin.core.execution.ray.common.task_wrapper.RayTask.deploy` method of :py:class:`~modin.core.execution.ray.common.task_wrapper.RayTask` where :py:class:`~modin.core.storage_formats.pandas.parsers.PandasCSVParser` parses data
on each single Ray worker. After remote tasks are finished, additional result postprocessing is performed,
and new query compiler with data read is returned.

In the event of writing data to a CSV file, for example, :py:class:`~modin.core.execution.ray.implementations.pandas_on_ray.io.PandasOnRayIO` processes
the user query to execute it on Ray workers. Then, :py:class:`~modin.core.execution.ray.implementations.pandas_on_ray.io.PandasOnRayIO` appeals to
:py:class:`~modin.core.execution.ray.implementations.pandas_on_ray.io.PandasOnRayDataframe` to be decomposed into row-wise partitions
that will be written into the file in parallel in Ray workers.