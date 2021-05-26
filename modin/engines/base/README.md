## Implementation Note

### Object Hierarchy

- `frame/partition.py` contains `PandasFramePartition` interface and its implementations.
- `frame/partition_manager.py` contains `PandasFramePartitionManager` interface and its implementations.
	- `PandasFramePartitionManager` manages 2D-array of `PandasFramePartition` object
- `frame/axis_partition.py` contains `BaseFrameAxisPartition` and with the following hierarchy:
	```
	BaseFrameAxisPartition -> PandasOnRayFrameAxisPartition -> {PandasOnRayFrameColumnPartition, PandasOnRayFrameRowPartition}
	```
	- `BaseFrameAxisPartition` is a high level view onto PandasFramePartitionManager' data. It is more
	   convenient to operate on `BaseFrameAxisPartition` sometimes.
