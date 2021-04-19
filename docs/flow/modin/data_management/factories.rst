Factories Module Description
""""""""""""""""""""""""""""

Brief description
'''''''''''''''''
Modin has many execution backends, the trace of any DataFrame API function call will end up in some backend-specific method. Factories module brings to route IO function calls from the API level to its actual backend-specific implementations.

Backend representation via Factories
''''''''''''''''''''''''''''''''''''
Backend — is a combination of the `QueryCompiler` and `Execution engine`. For example, ``PandasOnRay`` backend means the combination of the ``PandasQueryCompiler`` and ``Ray`` engine. 

In the scope of this module, each backend represented with a factory class located in ``factories.py``. Factory contains the IO module of the corresponding backend and responsible for dispatching calls of IO functions to their actual implementations in an underlying IO module. (For more information about IO module visit :doc:`related doc </flow/modin/engines/base/io.rst>`.)

Engine dispatcher
'''''''''''''''''
`EngineDispatcher` provides public methods whose interface corresponds to the same pandas IO functions, the only difference is that they return QueryCompiler of the selected backend instead of DataFrame. Engine dispatcher is responsible for routing these IO calls to the factory which represents the selected backend.

So when you call ``.read_csv`` function and your backend is `PandasOnRay` then the trace would be the following:
``modin.pandas.read_csv`` calls ``EngineDispatcher.read_csv``, which call ``.read_csv`` function of the factory of the selected backend, in our case it's ``PandasOnRayFactory._read_csv``, which forwards this call to the actual implementation of ``read_csv`` — to the ``PandasOnRayIO.read_csv``.
