:orphan:

ExperimentalPandasOnDask Execution
==================================

`ExperimentalPandasOnDask` execution keeps the underlying mechanisms of :doc:`PandasOnDask </flow/modin/core/execution/dask/implementations/pandas_on_dask/index>`
execution architecturally unchanged and adds experimental features of ``Data Transformation``, ``Data Ingress`` and ``Data Egress`` (e.g. :py:func:`~modin.experimental.pandas.read_pickle_distributed`).

PandasOnDask and ExperimentalPandasOnDask differences
-----------------------------------------------------

- another Factory ``PandasOnDaskFactory`` -> ``ExperimentalPandasOnDaskFactory``
- another IO class ``PandasOnDaskIO`` -> ``ExperimentalPandasOnDaskIO``

ExperimentalPandasOnDaskIO classes and modules
----------------------------------------------

- :py:class:`~modin.experimental.core.execution.dask.implementations.pandas_on_dask.io.io.ExperimentalPandasOnDaskIO`
- :py:class:`~modin.core.execution.dispatching.factories.factories.ExperimentalPandasOnDaskFactory`
- :py:class:`~modin.experimental.core.io.text.csv_glob_dispatcher.ExperimentalCSVGlobDispatcher`
- :py:class:`~modin.experimental.core.io.sql.sql_dispatcher.ExperimentalSQLDispatcher`
- :py:class:`~modin.experimental.core.io.pickle.pickle_dispatcher.ExperimentalPickleDispatcher`
- :py:class:`~modin.experimental.core.io.text.custom_text_dispatcher.ExperimentalCustomTextDispatcher`
- :py:class:`~modin.core.storage_formats.pandas.parsers.PandasCSVGlobParser`
- :doc:`ExperimentalPandasOnDask IO module </flow/modin/experimental/core/execution/dask/implementations/pandas_on_dask/io/index>`
