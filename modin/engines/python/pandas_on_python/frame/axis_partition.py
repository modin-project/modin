from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import pandas

from modin.engines.base.frame.axis_partition import PandasOnXFrameFullAxisPartition
from .partition import PandasOnPythonFramePartition


class PandasOnPythonFrameFullAxisPartition(PandasOnXFrameFullAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        self.list_of_blocks = [obj.data for obj in list_of_blocks]

    partition_type = PandasOnPythonFramePartition
    instance_type = pandas.DataFrame


class PandasOnPythonFrameFullColumnPartition(PandasOnPythonFrameFullAxisPartition):
    """The column partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnPythonFrameFullRowPartition(PandasOnPythonFrameFullAxisPartition):
    """The row partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1
