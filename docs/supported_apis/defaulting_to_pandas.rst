Defaulting to pandas
====================

Currently Modin does not support distributed execution for all methods from pandas API.
The remaining unimplemented methods are being executed in a mode called "default to pandas".
This allows users to continue using Modin even though their workloads contain functions not yet implemented in Modin.
Here is a diagram of how we convert to pandas and perform the operation:

.. image:: /img/convert_to_pandas.png
   :align: center

We first convert to a pandas DataFrame, then perform the operation. **There is a
performance penalty for going from a partitioned Modin DataFrame to pandas because of
the communication cost and single-threaded nature of pandas.** Once the pandas operation
has completed, we convert the DataFrame back into a partitioned Modin DataFrame. This
way, operations performed after something defaults to pandas will be optimized with
Modin.

The exact methods we have implemented are listed in the respective subsections:

* :doc:`DataFrame </supported_apis/dataframe_supported>`
* :doc:`Series </supported_apis/series_supported>`
* :doc:`utilities </supported_apis/utilities_supported>`
* :doc:`I/O </supported_apis/io_supported>`

We have taken a community-driven approach to implementing new methods. We did a `study
on pandas usage`_ to learn what the most-used APIs are. Modin currently supports **93%**
of the pandas API based on our study of pandas usage, and we are actively expanding the
API.
**To request implementation, file an issue at https://github.com/modin-project/modin/issues
or send an email to feature_requests@modin.org.**

.. _`study on pandas usage`: https://github.com/modin-project/study_kaggle_usage
