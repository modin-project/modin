Installation
============

There are a couple of ways to install Modin. Most users will want to install with
``pip``, but some users may want to build from the master branch on the `GitHub repo`_.
The master branch has the most recent patches, but may be less stable than a release
installed from ``pip``.

Installing with pip
-------------------

Stable version
""""""""""""""

Modin can be installed with pip. To install the most recent stable release run the following:

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

Before most major releases, we will upload a release candidate to If you would like to
install a pre-release of Modin, run the following:

.. code-block:: bash

  pip install --pre modin

These pre-releases are uploaded for dependencies and users to test their existing code
to ensure that it still works. If you find something wrong, please raise an issue_ or
email the bug reporter: bug_reports@modin.org.

Installing specific dependency sets
"""""""""""""""""""""""""""""""""""

Modin has a number of specific dependency sets for running Modin on different backends
or for different functionalities of Modin. Here is a list of dependency sets for Modin:

.. code-block:: bash

  pip install "modin[dask]" # If you want to use the Dask backend

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

For installation on Windows, we recommend using the Dask_ Engine. Ray does not support Windows,
so it will not be possible to install ``modin[ray]`` or ``modin[all]``. It is possible to use
Windows Subsystem For Linux (WSL_), but this is generally not recommended due to the limitations
and poor performance of Ray on WSL, a roughly 2-3x cost. To install with the Dask_ engine, run the
following using ``pip``:

.. code-block:: bash

    pip install modin[dask]

You may already have a recent version of Dask_ installed, in which case you can simply ``pip install modin``.

Building Modin from Source
--------------------------

If you're planning on contributing_ to Modin, you will need to ensure that you are
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
.. _`out of core`: out_of_core.html
.. _WSL: https://docs.microsoft.com/en-us/windows/wsl/install-win10
.. _Ray: http://ray.readthedocs.io
.. _contributing: contributing.html
.. _Dask: https://github.com/dask/dask
