:orphan:

ExperimentalPandasOnRay Execution
=================================

`ExperimentalPandasOnRay` execution keeps the underlying mechanisms of :doc:`PandasOnRay </flow/modin/core/execution/ray/implementations/pandas_on_ray/index>`
execution architecturally unchanged and adds experimental features of ``Data Transformation``, ``Data Ingress`` and ``Data Egress`` (e.g. :py:func:`~modin.experimental.pandas.read_pickle_distributed`).

PandasOnRay and ExperimentalPandasOnRay differences
---------------------------------------------------

- another Factory ``PandasOnRayFactory`` -> ``ExperimentalPandasOnRayFactory``
- another IO class ``PandasOnRayIO`` -> ``ExperimentalPandasOnRayIO``

ExperimentalPandasOnRayIO classes and modules
---------------------------------------------

- :py:class:`~modin.experimental.core.execution.ray.implementations.pandas_on_ray.io.io.ExperimentalPandasOnRayIO`
- :py:class:`~modin.core.execution.dispatching.factories.factories.ExperimentalPandasOnRayFactory`
- :py:class:`~modin.experimental.core.io.text.csv_glob_dispatcher.ExperimentalCSVGlobDispatcher`
- :py:class:`~modin.experimental.core.io.sql.sql_dispatcher.ExperimentalSQLDispatcher`
- :py:class:`~modin.experimental.core.io.pickle.pickle_dispatcher.ExperimentalPickleDispatcher`
- :py:class:`~modin.experimental.core.io.text.custom_text_dispatcher.ExperimentalCustomTextDispatcher`
- :py:class:`~modin.core.storage_formats.pandas.parsers.PandasCSVGlobParser`
- :doc:`ExperimentalPandasOnRay IO module </flow/modin/experimental/core/execution/ray/implementations/pandas_on_ray/io/index>`
