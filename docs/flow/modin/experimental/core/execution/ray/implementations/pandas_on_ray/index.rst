:orphan:

Experimental PandasOnRay Execution
==================================

Experimental PandasOnRay execution is designed to leave the underlying mechanisms of PandasOnRay
execution: "Data Transformation", "Data Ingress", "Data Egress" architecturally
:doc:`unchanged </flow/modin/core/execution/ray/implementations/pandas_on_ray/index>`.

Main differencies between these two executions
----------------------------------------------
- new Factory ``PandasOnRayFactory`` -> ``ExperimentalPandasOnRayFactory``
- new IO class ``PandasOnRayIO`` -> ``ExperimentalPandasOnRayIO``

Objects implementations for Experimental PandasOnRay IO
-------------------------------------------------------
- :py:class:`~modin.experimental.core.execution.ray.implementations.pandas_on_ray.io.ExperimentalPandasOnRayIO`
- :py:class:`~modin.core.execution.dispatching.factories.factories.ExperimentalPandasOnRayFactory`
- :py:class:`~modin.core.io.text.csv_glob_dispatcher.CSVGlobDispatcher`
- :py:class:`~modin.core.storage_formats.pandas.parsers.PandasCSVGlobParser`

Related
-------
- :doc:`ExperimentalPandasOnRayIO </flow/modin/experimental/core/execution/ray/implementations/pandas_on_ray/io/index>`
