Advanced Usage
==============

.. toctree::
   :titlesonly:
   :hidden:

   /flow/modin/distributed/dataframe/pandas
   spreadsheets_api
   progress_bar
   modin_xgboost
   modin_logging
   batch
   modin_engines

.. meta::
    :description lang=en:
        Description of Modin's advanced features.

Modin aims to not only optimize pandas, but also provide a comprehensive,
integrated toolkit for data scientists. We are actively developing data science tools
such as DataFrame spreadsheet integration, DataFrame algebra, progress bars, SQL queries
on DataFrames, and more. Join us on `Slack`_ for the latest updates!

Modin engines
-------------

Modin supports a series of execution engines such as Ray_, Dask_, `MPI through unidist`_,
each of which might be a more beneficial choice for a specific scenario. When doing the first operation
with Modin it automatically initializes one of the engines to further perform distributed/parallel computation.
If you are familiar with a concrete execution engine, it is possible to initialize the engine on your own and
Modin will automatically attach to it. Refer to :doc:`Modin engines </usage_guide/advanced_usage/modin_engines>` page
for more details.

Additional APIs
---------------

Modin also supports these additional APIs on top of pandas to improve user experience.

- :py:meth:`~modin.pandas.DataFrame.modin.to_pandas` -- convert a Modin DataFrame/Series to a pandas DataFrame/Series.
- :py:func:`~modin.pandas.io.from_pandas` -- convert a pandas DataFrame to a Modin DataFrame.
- :py:meth:`~modin.pandas.DataFrame.modin.to_ray` -- convert a Modin DataFrame/Series to a Ray Dataset.
- :py:func:`~modin.pandas.io.from_ray` -- convert a Ray Dataset to a Modin DataFrame.
- :py:meth:`~modin.pandas.DataFrame.modin.to_dask` -- convert a Modin DataFrame/Series to a Ray Dataset.
- :py:func:`~modin.pandas.io.from_dask` -- convert a Modin DataFrame/Series to a Dask DataFrame/Series.
- :py:func:`~modin.pandas.io.from_map` -- create a Modin DataFrame from map function applied to an iterable object.
- :py:func:`~modin.pandas.io.from_arrow` -- convert an Arrow Table to a Modin DataFrame.
- :py:func:`~modin.experimental.pandas.read_csv_glob` -- read multiple files in a directory.
- :py:func:`~modin.experimental.pandas.read_sql` -- add optional parameters for the database connection.
- :py:func:`~modin.experimental.pandas.read_custom_text` -- read custom text data from file.
- :py:func:`~modin.experimental.pandas.read_pickle_glob`  -- read multiple pickle files in a directory.
- :py:func:`~modin.experimental.pandas.read_parquet_glob`  -- read multiple parquet files in a directory.
- :py:func:`~modin.experimental.pandas.read_json_glob`  -- read multiple json files in a directory.
- :py:func:`~modin.experimental.pandas.read_xml_glob`  -- read multiple xml files in a directory.
- :py:meth:`~modin.pandas.DataFrame.modin.to_pickle_glob` -- write to multiple pickle files in a directory.
- :py:meth:`~modin.pandas.DataFrame.modin.to_parquet_glob` -- write to multiple parquet files in a directory.
- :py:meth:`~modin.pandas.DataFrame.modin.to_json_glob` -- write to multiple json files in a directory.
- :py:meth:`~modin.pandas.DataFrame.modin.to_xml_glob` -- write to multiple xml files in a directory.

DataFrame partitioning API
--------------------------

Modin DataFrame provides an API to directly access partitions: you can extract physical partitions from
a :py:class:`~modin.pandas.dataframe.DataFrame`, modify their structure by reshuffling or applying some
functions, and create a DataFrame from those modified partitions. Visit
:doc:`pandas partitioning API </flow/modin/distributed/dataframe/pandas>` documentation to learn more.

Modin Spreadsheet API
---------------------

The Spreadsheet API for Modin allows you to render the dataframe as a spreadsheet to easily explore
your data and perform operations on a graphical user interface. The API also includes features for recording
the changes made to the dataframe and exporting them as reproducible code. Built on top of Modin and SlickGrid,
the spreadsheet interface is able to provide interactive response times even at a scale of billions of rows.
See our `Modin Spreadsheet API documentation`_ for more details.

.. figure:: /img/modin_spreadsheet_mini_demo.gif
   :align: center
   :width: 650px
   :height: 350px

Progress Bar
------------

Visual progress bar for Dataframe operations such as groupby and fillna, as well as for file reading operations such as
read_csv. Built using the `tqdm`_ library and Ray execution engine. See `Progress Bar documentation`_ for more details.

.. figure:: /img/progress_bar_example.png
   :align: center

Dataframe Algebra
-----------------

A minimal set of operators that can be composed to express any dataframe query for use in query planning and optimization.
See our `paper`_ for more information, and full documentation is coming soon!

Distributed XGBoost on Modin
----------------------------

Modin provides an implementation of `distributed XGBoost`_ machine learning algorithm on Modin DataFrames. See our
:doc:`Distributed XGBoost on Modin documentation <modin_xgboost>` for details about installation and usage, as well as
:doc:`Modin XGBoost architecture documentation </flow/modin/experimental/xgboost>` for information about implementation and
internal execution flow.

Logging with Modin
------------------

Modin logging offers users greater insight into their queries by logging internal Modin API calls, partition metadata,
and system memory. Logging is disabled by default, but when it is enabled, log files are written to a local `.modin` directory
at the same directory level as the notebook/script used to run Modin. See our :doc:`Logging with Modin documentation <modin_logging>`
for usage information.

Batch Pipeline API
------------------
Modin provides an experimental batched API that pipelines row parallel queries. See our :doc:`Batch Pipline API Usage Guide <batch>`
for a walkthrough on how to use this feature, as well as :doc:`Batch Pipeline API documentation </flow/modin/experimental/batch>`
for more information about the API.

Fuzzydata Testing
-----------------

An experimental GitHub Action on pull request has been added to Modin, which automatically runs the Modin codebase against
`fuzzydata`, a random dataframe workflow generator. The resulting workflow that was used to test Modin codebase can be
downloaded as an artifact from the GitHub Actions tab for further inspection. See `fuzzydata`_ for more details.

.. _`Modin Spreadsheet API documentation`: spreadsheets_api.html
.. _`Progress Bar documentation`: progress_bar.html
.. _`Paper`: https://arxiv.org/pdf/2001.00888.pdf
.. _`Slack`: https://modin.org/slack.html
.. _`tqdm`: https://github.com/tqdm/tqdm
.. _`distributed XGBoost`: https://medium.com/intel-analytics-software/distributed-xgboost-with-modin-on-ray-fc17edef7720
.. _`fuzzydata`: https://github.com/suhailrehman/fuzzydata
.. _Ray: https://github.com/ray-project/ray
.. _Dask: https://github.com/dask/distributed
.. _`MPI through unidist`: https://github.com/modin-project/unidist
