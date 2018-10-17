Installation
============

There are a couple of ways to start using Pandas on Ray. Most users will want to
install with ``pip``, but some users may want to build from the master branch on
the `GitHub repo`_.

Installing with pip
-------------------

Modin can be installed with pip.

.. code-block:: bash

  pip install modin

Installing from the GitHub master branch
----------------------------------------

If you'd like to try Modin using the most recent updates from the master branch, you can
also use `pip`.

.. code-block:: bash

  pip install git+https://github.com/modin-project/modin

This will install directly from the repo without you having to clone it! Please be aware
that these changes have not made it into a release and may not be completely stable.

Windows
-------

For installation on Windows, we recommend using Windows Subsystem for Linux (WSL_). This
will allow you to use Linux commands on your Windows machine.

One of our dependencies is Ray_. Ray is not yet supported natively on Windows, so in
order to install it you need to use the WSL if you are on Windows.

Once you've done installed WSL and you run the WSL application, you can install Modin
just like you would on Linux or Mac:

.. code-block:: bash

    pip install modin

Once you've done this, Modin will be installed. However, it is important to note that
you must execute `python`, `ipython` and `jupyter` from the WSL application.


Dependencies
------------

Currently, Modin depends on pandas version 0.23.4. The API of pandas has a
tendency to change some with each release, so we pin our current version to the
most recent version to take advantage of the newest additions. This also
typically means better performance and more correct code.

Modin also depends on Ray_. Ray is a task-parallel execution framework for
parallelizing new and existing applications with minor code changes. Currently,
we depend on the most recent Ray release: 0.5.3.

Building Modin from Source
--------------------------

If you're planning on contributing_ to Modin, you will need to ensure that you are
building Modin from the local repository that you are working off of. Occassionally,
there are issues in overlapping Modin installs from pypi and from source. To avoid these
issues, we recommend uninstalling Modin before you install from source:

.. code-block:: bash

  pip uninstall modin

To build from source, you first must clone the repo:

.. code-block:: bash

  git clone https://github.com/modin-project/modin.git

Once cloned, ``cd`` into the ``modin`` directory and use ``pip`` to install:

.. code-block:: bash

  cd modin
  pip install -e .

.. _`GitHub repo`: https://github.com/modin-project/modin/tree/master
.. _WSL: https://docs.microsoft.com/en-us/windows/wsl/install-win10
.. _Ray: http://ray.readthedocs.io
.. _contributing: contributing.html
