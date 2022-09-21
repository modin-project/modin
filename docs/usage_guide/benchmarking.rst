Benchmarking Modin
==================

Summary
-------
To benchmark a single Modin function, often turning on the
:doc:`configuration variable </flow/modin/config>` variable
:code:`BenchmarkMode` will suffice.

There is no simple way to benchmark more complex Modin workflows, though
benchmark mode or calling ``repr`` on Modin objects may be useful. The
:doc:`Modin logs </usage_guide/advanced_usage/modin_logging>` may help you
identify bottlenecks in your code, and they may also help profile the execution
of each Modin function.

Modin's execution and benchmark mode
------------------------------------

Most of Modin's execution happens asynchronously, i.e. in separate processes that run
independently of the main program flow. Some execution is also lazy, meaning that it
doesn't start immediately once the user calls a Modin function. While Modin provides
the same API as pandas, lazy and asynchronous execution can often make it hard to
tell how much time each Modin function call takes, as well as to compare Modin's
performance to pandas and other similar libraries.

.. note::
    All examples in this doc use the system specified at the bottom of this page.

Consider the following ipython script:

.. code-block:: python

    import modin.pandas as pd
    from modin.config import MinPartitionSize
    import time
    import ray

    ray.init()
    df = pd.DataFrame(list(range(MinPartitionSize.get())))
    %time result = df.applymap(lambda x: time.sleep(0.2) or x)
    %time print(result)


Modin takes just 2.68 milliseconds for the ``applymap``, and 6.78 seconds to print
the result. However, if we run this script in pandas by replacing
:code:`import modin.pandas as pd` with :code:`import pandas as pd`, the ``applymap``
takes 6.5 seconds, and printing the result takes just 3.36 milliseconds.

Both pandas and Modin start executing the ``applymap`` as soon as the interpreter
evalutes it. While pandas blocks until the ``applymap`` has finished, Modin just kicks
off asynchronous functions in remote ray processes. Printing the function result
is fairly fast in pandas and Modin, but before Modin can print the data, it has to
wait until all the remote functions complete.

To time how long Modin takes for a single operation, you should typically use
benchmark mode. You can turn on benchmark mode on at any point as follows:

.. code-block:: python

    from modin.config import BenchmarkMode
    BenchmarkMode.put(True)

Rerunning the script above with benchmark mode on, the Modin ``applymap`` takes
6.79 seconds, and the ``print`` takes 5.46 milliseconds. These timings better
reflect where Modin is spending its execution time.

A caveat about benchmark mode
-----------------------------

While benchmark code is often good for measuring the performance of a single
Modin function call, it can underestimate Modin's performance in cases where
Modin's asynchronous execution improves Modin's performance. Consider the
following script with benchmark mode on:

.. code-block:: python

    import numpy as np
    import time
    import ray
    from io import BytesIO

    import modin.pandas as pd
    from modin.config import BenchmarkMode, MinPartitionSize

    BenchmarkMode.put(True)

    start = time.time()
    df = pd.DataFrame(list(range(MinPartitionSize.get())), columns=['A'])
    result1 = df.applymap(lambda x: time.sleep(0.2) or x + 1)
    result2 = df.applymap(lambda x: time.sleep(0.2) or x + 2)
    result1.to_parquet(BytesIO())
    result2.to_parquet(BytesIO())
    end = time.time()
    print(f'applymap and write to parquet took {end - start} seconds.')

.. code-block::python

The script does two slow ``applymap`` on a dataframe and then writes each result
to a buffer. The whole script takes 13 seconds with benchmark mode on, but
just 7 seconds with benchmark mode off. Because Modin can run the ``applymap``
asynchronously, it can start writing the first result to its buffer while
it's still computing the second result. With benchmark mode on, Modin has to
execute every function synchronously instead.

How to benchmark complex workflows
----------------------------------

Typically, to benchmark Modin's overall performance on your workflow, you
should start by looking at end-to-end performance with benchmark mode off.
It's common for Modin worfklows to end with writing results to one or more
files, or with printing some Modin objects to an interactive console. Such
end points are natural ways to make sure that all of the Modin execution that
you require is complete.

To measure more fine-grained performance, it can be helpful to turn
benchmark mode on, but beware that doing so may reduce your script's overall
performance and thus may not reflect where Modin is normally spending execution
time, as pointed out above.

Turning on :doc:`Modin logging </usage_guide/advanced_usage/modin_logging>` and
using the Modin logs can also help you profile your workflow. The Modin logs
can also give a detailed break down of the performance of each Modin function
at each Modin :doc:`layer </development/architecture>`. Log mode is more
useful when used in conjuction with benchmark mode.

Sometimes, if you don't have a natural end-point to your workflow, you can
just call ``repr`` on the workflow's final Modin objects. That will typically
block on any asynchronous computation. However, beware that ``repr`` can also
be misleading, e.g. here:

.. code-block:: python

    import time
    import ray
    from io import BytesIO

    import modin.pandas as pd
    from modin.config import MinPartitionSize, NPartitions

    MinPartitionSize.put(32)
    NPartitions.put(16)

    def slow_add_one(x):
      if x == 5000:
        time.sleep(10)
      return x + 1

    ray.init()
    df1 = pd.DataFrame(list(range(10_000)), columns=['A'])
    result = df1.applymap(slow_add_one)
    %time repr(result)
    # time.sleep(10)
    %time result.to_parquet(BytesIO())
.. code-block::python

The ``repr`` takes only 802 milliseconds, but writing the result to a buffer
takes 9.84 seconds. However, if you uncomment the :code:`time.sleep` before the
:code:`to_parquet` call, the :code:`to_parquet` takes just 23.8 milliseconds!
The problem is that the ``repr`` blocks only on getting the first few and the
last few rows, but the slow execution is for row 5001, which Modin is
computing asynchronously in the background even after ``repr`` finishes.

.. note::
    If you see any Modin documentation touting Modin's speed without using
    benchmark mode or otherwise guaranteeing that Modin is finishing all asynchronous
    and deferred computation, you should file an issue on the Modin GitHub. It's
    not fair to compare the speed of an async Modin function call to an equivalent
    synchronous call using another library.

Appendix: specs
---------------
The example scripts here were run on the following system:

- **OS Platform and Distribution (e.g., Linux Ubuntu 16.04)**: macOS Monterey 12.4
- **Modin version**: d6d503ac7c3028d871c34d9e99e925ddb0746df6
- **Ray version**: 2.0.0
- **Python version**: 3.10.4
- **Machine**: MacBook Pro (16-inch, 2019)
- **Processor**: 2.3 GHz 8-core Intel Core i9 processor
- **Memory**: 16 GB 2667 MHz DDR4
