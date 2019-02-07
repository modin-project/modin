from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas

from modin.engines.base.axis_partition import PandasOnXAxisPartition
from .remote_partition import PandasOnPythonRemotePartition


class PandasOnPythonAxisPartition(PandasOnXAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseRemotePartition object for ease of use
        self.list_of_blocks = [obj.data for obj in list_of_blocks]

    partition_type = PandasOnPythonRemotePartition
    instance_type = pandas.DataFrame


class PandasOnPythonColumnPartition(PandasOnPythonAxisPartition):
    """The column partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnPythonRowPartition(PandasOnPythonAxisPartition):
    """The row partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1
