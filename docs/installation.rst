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

Currently, Modin depends on pandas version 0.22. The API of pandas has a
tendency to change some with each release, so we pin our current version to the
most recent version to take advantage of the newest additions. This also
typically means better performance and more correct code.

Building Modin from Source
--------------------------

To build from source, you first must clone the repo:

.. code-block:: bash

  git clone https://github.com/modin-project/modin.git

Once cloned, ``cd`` into the ``modin`` directory and use ``pip`` to install:

.. code-block:: bash

  cd modin
  pip install -e .

.. _`GitHub repo`: https://github.com/modin-project/modin/tree/master
