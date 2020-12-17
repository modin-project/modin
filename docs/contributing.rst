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

Certificate of Origin
---------------------

To keep a clear track of who did what, we use a `sign-off` procedure (same requirements 
for using the signed-off-by process as the Linux kernel has 
https://www.kernel.org/doc/html/v4.17/process/submitting-patches.html) on patches or pull 
requests that are being sent. The sign-off is a simple line at the end of the explanation 
for the patch, which certifies that you wrote it or otherwise have the right to pass it 
on as an open-source patch. The rules are pretty simple: if you can certify the below:

CERTIFICATE OF ORIGIN V 1.1
^^^^^^^^^^^^^^^^^^^^^^^^^^^
"By making a contribution to this project, I certify that:

1.) The contribution was created in whole or in part by me and I have the right to
submit it under the open source license indicated in the file; or
2.) The contribution is based upon previous work that, to the best of my knowledge, is
covered under an appropriate open source license and I have the right under that license
to submit that work with modifications, whether created in whole or in part by me, under
the same open source license (unless I am permitted to submit under a different
license), as indicated in the file; or
3.) The contribution was provided directly to me by some other person who certified (a),
(b) or (c) and I have not modified it.
4.) I understand and agree that this project and the contribution are public and that a
record of the contribution (including all personal information I submit with it,
including my sign-off) is maintained indefinitely and may be redistributed consistent
with this project or the open source license(s) involved."


.. code-block:: bash

   This is my commit message

   Signed-off-by: Awesome Developer <developer@example.org>


Code without a proper signoff cannot be merged into the
master branch. Note: You must use your real name (sorry, no pseudonyms or anonymous
contributions.)

The text can either be manually added to your commit body, or you can add either ``-s``
or ``--signoff`` to your usual ``git commit`` commands:



.. code-block:: bash

   git commit --signoff
   git commit -s

This will use your default git configuration which is found in .git/config. To change
this, you can use the following commands:

.. code-block:: bash

   git config --global user.name "Awesome Developer"
   git config --global user.email "awesome.developer.@example.org"

If you have authored a commit that is missing the signed-off-by line, you can amend your
commits and push them to GitHub.

.. code-block:: bash

   git commit --amend --signoff

If you've pushed your changes to GitHub already you'll need to force push your branch
after this with ``git push -f``.

Commit Message formatting
-------------------------
To ensure that all commit messages in the master branch follow a specific format, we
enforce that all commit messages must follow the following format:

.. code-block:: bash

    FEAT-#9999: Add `DataFrame.rolling` functionality, to enable rolling window operations

The ``FEAT`` component represents the type of commit. This component of the commit
message can be one of the following:

* FEAT: A new feature that is added
* DOCS: Documentation improvements or updates
* FIX: A bugfix contribution
* REFACTOR: Moving or removing code without change in functionality
* TEST: Test updates or improvements

The ``#9999`` component of the commit message should be the issue number in the Modin
GitHub issue tracker: https://github.com/modin-project/modin/issues. This is important
because it links commits to their issues.

The commit message should follow a colon (:) and be descriptive and succinct.

Development Dependencies
------------------------

We recommend doing development in a virtualenv or conda environment, though this decision 
is ultimately yours. You will want to run the following in order to install all of the required
dependencies for running the tests and formatting the code:

.. code-block:: bash

  conda env create --file environment.yml
  # or
  pip install -r requirements.txt

Code Formatting and Lint
------------------------

We use black_ for code formatting. Before you submit a pull request, please make sure
that you run the following from the project root:

.. code-block:: bash

  black modin/

We also use flake8_ to check linting errors. Running the following from the project root
will ensure that it passes the lint checks on Github Actions:

.. code-block:: bash

  flake8 .

We test that this has been run on our `Github Actions`_ test suite. If you do this and find
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

Performance measurement
-----------------------

We use Asv_ tool for performance tracking of various Modin functionality.

Here are some scenarios in which Asv can be used:

* It is necessary to check the impact of the new patch on the performance of a certain set of operations:

.. code-block:: bash

  asv continuous -f 1.05 src/master HEAD -b TimeGroupBy --launch-method=spawn

* It is necessary to check presence of errors inside of benchmarks after making changes or writing new ones:

.. code-block:: bash

  asv run --quick --show-stderr --python=same --launch-method=spawn

* You just need to run the entire test suite to get the current time numbers:

.. code-block:: bash

  asv run --launch-method=spawn

* It is necessary to check the range of commits for performance degradation:

.. code-block:: bash

 asv run [start_hash]..[end_hash] --launch-method=spawn
 asv publish
 asv preview

For more consistent results, you may need to use the following parameters:

* ``-a sample_time=1``
* ``-a warmup_time=1``
* ``-a processes=4``

Some details about using Modin on Ray with Asv:

* ``--launch-method=forkserver`` is not working
* Each set of parameters for each test is launched in its own process, which brings
  a large overhead, since for each process redis server and other necessary binaries
  from ray initialization are started and destroyed.

Some details for maintenance:

* ``modin/asv_bench/asv.conf.json`` contains the modin dependencies, with which testing takes place.
We need to keep them up to date according to the dependencies in ``setup.py``.


Building documentation
----------------------

To build the documentation, please follow the steps below from the project root:

.. code-block:: bash

    cd docs
    pip install -r requirements-doc.txt
    sphinx-build -b html . build

To visualize the documentation locally, run the following from `build` folder:

.. code-block:: bash

    python -m http.server <port>
    # python -m http.server 1234

then open the browser at `0.0.0.0:<port>` (e.g. `0.0.0.0:1234`).

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
.. _Github Actions: https://github.com/features/actions
.. _testing:
.. _Asv: https://github.com/airspeed-velocity/asv#airspeed-velocity
.. _developer mailing list: https://groups.google.com/forum/#!forum/modin-dev
