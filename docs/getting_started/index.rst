Getting Started
===============

.. toctree::
    :titlesonly:
    :hidden:
    
    using_modin
    out_of_core
    pandas
    dask
    faq
    troubleshooting

.. meta::
    :description lang=en:
        Introduction to Modin.

There are several ways to install Modin. Most users will want to install with
``pip`` or using ``conda`` tool, but some users may want to build from the master branch
on the `GitHub repo`_. The master branch has the most recent patches, but may be less
stable than a release installed from ``pip`` or ``conda``.

Installing with pip
-------------------

Stable version
""""""""""""""

Modin can be installed with ``pip`` on Linux, Windows and MacOS. 2 engines are available for those platforms:
:doc:`Ray</developer/using_pandas_on_ray>` and :doc:`Dask</developer/using_pandas_on_dask>`
To install the most recent stable release run the following:

.. code-block:: bash

  pip install -U modin # -U for upgrade in case you have an older version

If you don't have Ray_ or Dask_ installed, you will need to install Modin with one of the targets:

.. code-block:: bash

  pip install modin[ray] # Install Modin dependencies and Ray to run on Ray
  pip install modin[dask] # Install Modin dependencies and Dask to run on Dask
  pip install modin[all] # Install all of the above

Modin will automatically detect which engine you have installed and use that for
scheduling computation!

Release candidates
""""""""""""""""""

Before most major releases, we will upload a release candidate to test and check if there are any problems. If you would like to install a pre-release of Modin, run the following:

.. code-block:: bash

  pip install --pre modin

These pre-releases are uploaded for dependencies and users to test their existing code
to ensure that it still works. If you find something wrong, please raise an issue_ or
email the bug reporter: bug_reports@modin.org.

Installing specific dependency sets
"""""""""""""""""""""""""""""""""""

Modin has a number of specific dependency sets for running Modin on different execution engines and
storage formats or for different functionalities of Modin. Here is a list of dependency sets for Modin:

.. code-block:: bash

  pip install "modin[dask]" # If you want to use the Dask execution engine

Installing on Google Colab
"""""""""""""""""""""""""""

Modin can be used with Google Colab_ via the ``pip`` command, by running the following code in a new cell:

.. code-block:: bash

  !pip install modin[all]

Since Colab preloads several of Modin's dependencies by default, we need to restart the Colab environment once Modin is installed by either clicking on the :code:`"RESTART RUNTIME"` button in the installation output or by run the following code:

.. code-block:: python

  # Post-install automatically kill and restart Colab environment 
  import os
  os.kill(os.getpid(), 9)

Once you have restarted the Colab environment, you can use Modin in Colab in subsequent sessions.

Note that on the free version of Colab, there is a `limit on the compute resource <https://research.google.com/colaboratory/faq.html>`_. To leverage the full power of Modin, you may have to upgrade to Colab Pro to get access to more compute resources.

Installing with conda
---------------------

Using conda-forge channel
"""""""""""""""""""""""""

Modin releases can be installed using ``conda`` from conda-forge channel. Starting from 0.10.1
it is possible to install modin with chosen engine(s) alongside. Current options are:

+---------------------------------+---------------------------+-----------------------------+
| **Package name in conda-forge** | **Engine(s)**             | **Supported OSs**           |
+---------------------------------+---------------------------+-----------------------------+
| modin                           | Dask_                     |   Linux, Windows, MacOS     |
+---------------------------------+---------------------------+-----------------------------+
| modin-dask                      | Dask                      |   Linux, Windows, MacOS     |
+---------------------------------+---------------------------+-----------------------------+
| modin-ray                       | Ray_                      |       Linux, Windows        |
+---------------------------------+---------------------------+-----------------------------+
| modin-omnisci                   | OmniSci_                  |          Linux              |
+---------------------------------+---------------------------+-----------------------------+
| modin-all                       | Dask, Ray, OmniSci        |          Linux              |
+---------------------------------+---------------------------+-----------------------------+

So for installing Dask and Ray engines into conda environment following command should be used:

.. code-block:: bash

  conda install -c conda-forge modin-ray modin-dask

All set of engines could be available in conda environment by specifying

.. code-block:: bash

  conda install -c conda-forge modin-all

or explicitly

.. code-block:: bash

  conda install -c conda-forge modin-ray modin-dask modin-omnisci

Using Intel\ |reg| Distribution of Modin
""""""""""""""""""""""""""""""""""""""""

With ``conda`` it is also possible to install Intel Distribution of Modin, a special version of Modin 
that is part of Intel\ |reg| oneAPI AI Analytics Toolkit. This version of Modin is powered by :doc:`OmniSci</developer/using_omnisci>` 
engine that contains a bunch of optimizations for Intel hardware. More details can be found on `Intel Distribution of Modin`_ page.

Installing from the GitHub master branch
----------------------------------------

If you'd like to try Modin using the most recent updates from the master branch, you can
also use ``pip``.

.. code-block:: bash

  pip install git+https://github.com/modin-project/modin

This will install directly from the repo without you having to manually clone it! Please be aware
that these changes have not made it into a release and may not be completely stable.

Windows
-------

All Modin engines except :doc:`OmniSci</developer/using_omnisci>` are available both on Windows and Linux as mentioned above.
Default engine on Windows is :doc:`Ray</developer/using_pandas_on_ray>`.
It is also possible to use Windows Subsystem For Linux (WSL_), but this is generally not recommended due to the limitations
and poor performance of Ray on WSL, a roughly 2-3x cost. 

Building Modin from Source
--------------------------

If you're planning on :doc:`contributing </developer/contributing>` to Modin, you will need to ensure that you are
building Modin from the local repository that you are working off of. Occasionally,
there are issues in overlapping Modin installs from pypi and from source. To avoid these
issues, we recommend uninstalling Modin before you install from source:

.. code-block:: bash

  pip uninstall modin

To build from source, you first must clone the repo. We recommend forking the repository first
through the GitHub interface, then cloning as follows:

.. code-block:: bash

  git clone https://github.com/<your-github-username>/modin.git

Once cloned, ``cd`` into the ``modin`` directory and use ``pip`` to install:

.. code-block:: bash

  cd modin
  pip install -e .

.. _`GitHub repo`: https://github.com/modin-project/modin/tree/master
.. _issue: https://github.com/modin-project/modin/issues
.. _WSL: https://docs.microsoft.com/en-us/windows/wsl/install-win10
.. _Ray: http://ray.readthedocs.io
.. _Dask: https://github.com/dask/dask
.. _OmniSci: https://www.omnisci.com/platform/omniscidb
.. _`Intel Distribution of Modin`: https://software.intel.com/content/www/us/en/develop/tools/oneapi/components/distribution-of-modin.html#gs.86stqv
.. |reg|    unicode:: U+000AE .. REGISTERED SIGN
.. _Colab: https://colab.research.google.com/
