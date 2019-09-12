Pandas on Ray Optimizations
===========================

Pandas on Ray is the component of Modin that runs on the Ray execution Framework.
Currently, the in-memory format for Pandas on Ray is a pandas `DataFrame`_ on each
partition. There are a number of Ray-specific optimizations we perform, which are
explained below. Currently, Ray is the only execution framework supported on Modin.
There are additional optimizations we can do on the pandas in-memory format. Those are
also explained below.

Ray-specific optimizations
--------------------------

Ray_ is a high-performance task-parallel execution framework with Python and Java APIs.
It uses the plasma store and serialization formats of `Apache Arrow`_.

Normally, in order to start a Ray cluster, a user would have to use some of Ray's
command line tools or call ``ray.init``. Modin will automatically call ``ray.init`` for
users who are running on a single node. Otherwise a Ray cluster must be setup before
calling ``import modin.pandas as pd``. More about running Modin in a cluster can be
found in the `using Modin`_ documentation.

**Serialization of tasks and parameters**

The optimization that improves the performance the most is the pre-serialization of the
tasks and parameters. This is primarily applicable to map operations. We have designed
the system such that there is a single remote function that accepts a serialized
function as a parameter and applies it to a partition. The operation will be serialized
separately for each partition if we do not call ``ray.put`` on it first. The
``BaseFrameManager`` abstract class exposes a unified way to preprocess functions. The
primary purpose of the preprocess abstraction is to allow for optimizations such as
this.

**Memory Management**

The second optimization we perform is related to how Ray and Arrow handle memory.
Historically, pandas has used a significant amount of memory, and tends to create copies
even for some simple computations. The plasma store in Arrow is immutable, which can
cause problems for certain workloads, as objects that are no longer in scope for the
Python application can be kept around and consume memory in Arrow. To resolve this
issue, we free memory once the reference count for that memory goes to zero. This
component is still experimental, but we plan to keep iterating on it to make Modin as
memory efficient as possible.

Pandas-specific optimizations
-----------------------------

Pandas on Ray can take advantage of some of the properties of pandas in order to
optimize for both memory footprint and runtime.

**Indexing**

Internally, since each partition contains a pandas DataFrame, the indexing information
for both rows and columns would be duplicated for every partition. Because we use block
partitions layout, it would be replicated as many times as there were blocks. To avoid
this issue, we use a ``pandas.RangeIndex`` internally, which has a fixed memory cost.

This optimization is also used to determine which columns or rows were dropped during a
``dropna`` or other similar operation. We use the ``pandas.RangeIndex`` internal to the
partitions to communicate the missing values back to the external ``Index``.

.. _Dataframe: https://pandas.pydata.org/pandas-docs/version/0.23.4/generated/pandas.DataFrame.html
.. _Ray: https://github.com/ray-project/ray
.. _using Modin: using_modin.html
.. _Apache Arrow: https://arrow.apache.org
