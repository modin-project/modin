Modin
=====

*Scale your pandas workflows by changing one line of code*

.. raw:: html

    <p align="center"><b>To use Modin, replace the pandas import:</b></p>
  <embed>
    <a href="https://github.com/modin-project/modin"><img style="position: absolute; top: 0; right: 0; border: 0;" src="https://camo.githubusercontent.com/365986a132ccd6a44c23a9169022c0b5c890c387/68747470733a2f2f73332e616d617a6f6e6177732e636f6d2f6769746875622f726962626f6e732f666f726b6d655f72696768745f7265645f6161303030302e706e67" alt="Fork me on GitHub" data-canonical-src="https://s3.amazonaws.com/github/ribbons/forkme_right_red_aa0000.png"></a>
  </embed>

.. code-block:: python

  # import pandas as pd
  import modin.pandas as pd


Scale your pandas workflow by changing a single line of code.
-------------------------------------------------------------

Modin uses Ray_ to provide an effortless way to speed up your pandas notebooks, scripts,
and libraries. Unlike other distributed DataFrame libraries, Modin provides seamless
integration and compatibility with existing pandas code. Even using the DataFrame
constructor is identical.

.. code-block:: python

  import modin.pandas as pd
  import numpy as np

  frame_data = np.random.randint(0, 100, size=(2**10, 2**8))
  df = pd.DataFrame(frame_data)

To use Modin, you do not need to know how many cores your system has and you do not need
to  specify how to distribute the data. In fact, you can continue using your previous
pandas notebooks while experiencing a considerable speedup from Modin, even on a single
machine. Once you’ve changed your import statement, you’re ready to use Modin just like
you would pandas.

Faster pandas, even on your laptop
----------------------------------

.. image:: img/read_csv_benchmark.png
   :height: 350px
   :width: 300px
   :alt: Plot of read_csv
   :align: right

The ``modin.pandas`` DataFrame is an extremely light-weight parallel DataFrame. Modin
transparently distributes the data and computation so that all you need to do is
continue using the pandas API as you were before installing Modin. Unlike other parallel
DataFrame systems, Modin is an extremely light-weight, robust DataFrame. Because it is so
light-weight, Modin provides speed-ups of up to 4x on a laptop with 4 physical cores.

In pandas, you are only able to use one core at a time when you are doing computation of
any kind. With Modin, you are able to use all of the CPU cores on your machine. Even in
``read_csv``, we see large gains by efficiently distributing the work across your entire
machine.

.. code-block:: python

  import modin.pandas as pd

  df = pd.read_csv("my_dataset.csv")

Modin is a DataFrame for datasets from 1KB to 1TB+
--------------------------------------------------

We have focused heavily on bridging the solutions between DataFrames for small data
(e.g. pandas) and large data. Often data scientists require different tools for doing
the same thing on different sizes of data. The DataFrame solutions that exist for 1KB do
not scale to 1TB+, and the overheads of the solutions for 1TB+ are too costly for
datasets in the 1KB range. With Modin, because of its light-weight, robust, and scalable
nature, you get a fast DataFrame at 1KB and 1TB+.

**Modin is currently under active development. Requests and contributions are welcome!**


.. toctree::
   :maxdepth: 1
   :caption: Installation

   installation.rst

.. toctree::
   :maxdepth: 1
   :caption: Using Modin

   using_modin.rst
   pandas_supported.rst

.. toctree::
   :maxdepth: 1
   :caption: Contributing to Modin

   contributing.rst

.. toctree::
   :maxdepth: 1
   :caption: Implementation Details and Architecture

   architecture.rst
   pandas_on_ray.rst

.. toctree::
   :maxdepth: 1
   :caption: Help

   troubleshooting.rst
   contact.rst

.. toctree::
   :maxdepth: 1
   :caption: SQL on Ray

   sql_on_ray.rst

.. _Ray: https://github.com/ray-project/ray/
