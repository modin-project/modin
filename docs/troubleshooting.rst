Troubleshooting
===============

We hope your experience with Modin is bug-free, but there are some quirks about Modin
that may require troubleshooting.

Frequently encountered issues
-----------------------------

This is a list of the most frequently encountered issues when using Modin. Some of these
are working as intended, while others are known bugs that are being actively worked on.

``ArrowIOError: Broken Pipe``
"""""""""""""""""""""""""""""

One of the more frequently encountered issues is an ``ArrowIOError: Broken Pipe``. This
error can happen in a couple of different ways. One of the most common ways this is
encountered is from pressing ``CTRL + C`` sending a ``KeyboardInterrupt`` to Modin. In
Ray, when a ``KeyboardInterrupt`` is sent, Ray will shutdown. This causes the
``ArrowIOError: Broken Pipe`` because there is no longer an available plasma store for
working on remote tasks. This is working as intended, as it is not yet possible in Ray
to kill a task that has already started computation.

The other common way to encounter this ``Error`` is to let your computer go to sleep. As
an optimization, Ray will shutdown whenever the computer goes to sleep. This will result
in the same issue as above, because there is no longer a running instance of the plasma
store.

**Solution**

Retart your interpreter or notebook kernel.

**Avoiding this Error**

Avoid using ``KeyboardInterrupt`` and keeping your notebook or terminal running while
your machine is asleep.

Hanging on ``import modin.pandas as pd``
""""""""""""""""""""""""""""""""""""""""

This can happen when Ray fails to start. It will keep retrying, but often it is faster
to just restart the notebook or interpreter. Generally, this should not happen. Most
commonly this is encountered when starting multiple notebooks or interpreters in quick
succession.

**Solution**

Restart your interpreter or notebook kernel.

**Avoiding this Error**

Avoid starting many Modin notebooks or interpreters in quick succession. Wait 2-3
seconds before starting the next one.
