## Implementation Note

### Object Hierarchy

- `partitioning/partition.py` contains `PandasFramePartition` interface and its implementations.
- `partitioning/partition_manager.py` contains `PandasFramePartitionManager` interface and its implementations.
	- `PandasFramePartitionManager` manages 2D-array of `PandasFramePartition` object
- `partitioning/axis_partition.py` contains `BaseFrameAxisPartition` and with the following hierarchy:
	```
	BaseFrameAxisPartition -> PandasOnRayFrameAxisPartition -> {PandasOnRayFrameColumnPartition, PandasOnRayFrameRowPartition}
	```
	- `BaseFrameAxisPartition` is a high level view onto PandasFramePartitionManager' data. It is more
	   convenient to operate on `BaseFrameAxisPartition` sometimes.
