import pandas

from modin.engines.base.frame.axis_partition import PandasFrameAxisPartition
from .partition import PandasOnPythonFramePartition


class PandasOnPythonFrameAxisPartition(PandasFrameAxisPartition):
    def __init__(self, list_of_blocks):
        # Unwrap from BaseFramePartition object for ease of use
        for obj in list_of_blocks:
            obj.drain_call_queue()
        self.list_of_blocks = [obj.data for obj in list_of_blocks]

    partition_type = PandasOnPythonFramePartition
    instance_type = pandas.DataFrame


class PandasOnPythonFrameColumnPartition(PandasOnPythonFrameAxisPartition):
    """The column partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 0


class PandasOnPythonFrameRowPartition(PandasOnPythonFrameAxisPartition):
    """The row partition implementation for Ray. All of the implementation
        for this class is in the parent class, and this class defines the axis
        to perform the computation over.
    """

    axis = 1
