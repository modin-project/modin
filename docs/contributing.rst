Contributing
============

Getting Started
---------------

If you're interested in getting involved in the development of Modin, but aren't sure
where start, take a look at the issues tagged `Good first issue`_ or Documentation_.
These are issues that would be good for getting familiar with the codebase and better
understanding some of the more complex components of the architecture. There is
documentation here about the architecture_ that you will want to review in order to get
started.

Also, feel free to join the discussions on the `developer mailing list`_.

Development Dependencies
------------------------

We recommend doing development in a virtualenv, though this decision is ultimately
yours. You will want to run the following in order to install all of the required
dependencies for running the tests and formatting the code:

.. code-block:: bash

  pip install -U black flake8 pytest feather-format lxml openpyxl \
      xlrd numpy matplotlib --ignore-installed


Code Formatting and Lint
------------------------

We use black_ for code formatting. Before you submit a pull request, please make sure
that you run the following from the project root:

.. code-block:: bash

  black modin/

We also use flake8_ to check linting errors. Running the following from the project root
will ensure that it passes the lint checks on Travis:

.. code-block:: bash

  flake8 .

We test that this has been run on our `Travis CI`_ test suite. If you do this and find
that the tests are still failing, try updating your version of black and flake8.

Adding a test
-------------

If you find yourself fixing a bug or adding a new feature, don't forget to add a test to
the test suite to verify its correctness! More on testing and the layout of the tests
can be found in our testing_ documentation. We ask that you follow the existing
structure of the tests for ease of maintenance.

Running the tests
-----------------

To run the entire test suite, run the following from the project root:

.. code-block:: bash

  pytest modin/pandas/test

The test suite is very large, and may take a long time if you run every test. If you've
only modified a small amount of code, it may be sufficient to run a single test or some
subset of the test suite. In order to run a specific test run:

.. code-block:: bash

  pytest modin/pandas/test::test_new_functionality

The entire test suite is automatically run for each pull request.

Contributing a new execution framework or in-memory format
----------------------------------------------------------

If you are interested in contributing support for a new execution framework or in-memory
format, please make sure you understand the architecture_ of Modin.

The best place to start the discussion for adding a new execution framework or in-memory
format is the `developer mailing list`_.

More docs on this coming soon...

.. _Good first issue: https://github.com/modin-project/modin/issues?q=is%3Aissue+is%3Aopen+label%3A%22good+first+issue+%3Abeginner%3A%22
.. _Documentation: https://github.com/modin-project/modin/issues?q=is%3Aissue+is%3Aopen+label%3A%22documentation+%3Abookmark_tabs%3A%22
.. _architecture: architecture.html
.. _internal methods:
.. _black: https://github.com/ambv/black
.. _flake8: http://flake8.pycqa.org/en/latest/
.. _Travis CI: https://travis-ci.org/
.. _testing:
.. _developer mailing list: https://groups.google.com/forum/#!forum/modin-dev
