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
.. _Ray: http://ray.readthedocs.io
.. _contributing: contributing.html
