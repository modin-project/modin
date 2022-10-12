Progress Bar
============

The progress bar allows users to see the estimated progress and completion time of each line they run, 
in environments such as a shell or Jupyter notebook.

.. figure:: /img/progress_bar.gif
   :align: center

Quickstart
""""""""""

The progress bar uses the `tqdm` library to visualize displays:

.. code-block:: bash

   pip install tqdm


Import the progress bar into your notebook by running the following:


.. code-block:: python

    from modin.config import ProgressBar
    ProgressBar.enable()
