:orphan:

..
    TODO: add links to documentation for mentioned modules.

Factories Module Description
""""""""""""""""""""""""""""

Brief description
'''''''''''''''''
Modin has several execution engines and storage formats, combining them together forms certain backends. 
Calling any DataFrame API function will end up in some backend-specific method. The responsibility of dispatching high-level API calls to
backend-specific function belongs to the :ref:`QueryCompiler <query_compiler_def>`, which is determined at the time of the dataframe's creation by the factory of
the corresponding backend. The mission of this module is to route IO function calls from
the API level to its actual backend-specific implementations, which builds the
`QueryCompiler` of the appropriate backend.

Backend representation via Factories
''''''''''''''''''''''''''''''''''''
Backend is a combination of the :doc:`storage format </flow/modin/core/storage_formats/index>` and execution engine.
For example, ``PandasOnRay`` backend means the combination of the `pandas storage format` and `Ray` execution engine.

Each storage format has its own :ref:`Query Compiler <query_compiler_def>` which compiles the most efficient queries
for the corresponding :doc:`Modin Dataframe </flow/modin/core/dataframe/index>` implementation. Speaking about ``PandasOnRay``
backend, its Query Compiler is :doc:`PandasQueryCompiler </flow/modin/core/storage_formats/pandas/query_compiler>` and the
Dataframe implementation is :doc:`PandasDataframe </flow/modin/core/dataframe/pandas/dataframe>`,
which is general implementation for every backend of the  pandas storage format. The actual implementation of ``PandasOnRay`` frame
is defined by the :doc:`PandasOnRayDataframe </flow/modin/core/execution/ray/implementations/pandas_on_ray/dataframe>` class that
extends ``PandasDataframe``.

In the scope of this module, each backend is represented with a factory class located in
``modin/core/execution/dispatching/factories/factories.py``. Each factory contains a field that identifies the IO module of the corresponding backend. This IO module is
responsible for dispatching calls of IO functions to their actual implementations in the
underlying IO module. For more information about IO module visit :doc:`related doc </flow/modin/core/io/index>`.

Factory Dispatcher
''''''''''''''''''
The ``modin.core.execution.dispatching.factories.dispatcher.FactoryDispatcher`` class provides 
public methods whose interface corresponds to pandas IO functions, the only difference is that they return `QueryCompiler` of the
selected backend instead of DataFrame. ``FactoryDispatcher`` is responsible for routing
these IO calls to the factory which represents the selected backend.

So when you call ``read_csv()`` function and your backend is ``PandasOnRay`` then the
trace would be the following:

.. figure:: /img/factory_dispatching.svg
    :align: center

``modin.pandas.read_csv`` calls ``FactoryDispatcher.read_csv``, which calls ``.read_csv``
function of the factory of the selected backend, in our case it's ``PandasOnRayFactory._read_csv``,
which in turn forwards this call to the actual implementation of ``read_csv`` — to the
``PandasOnRayIO.read_csv``. The result of ``modin.pandas.read_csv`` will return a Modin
DataFrame with the appropriate `QueryCompiler` bound to it, which is responsible for
dispatching all of the further function calls.
