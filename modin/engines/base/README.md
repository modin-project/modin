## Implementation Note

### Object Hierarchy

- `remote_partition.py` contains `BaseFramePartition` interface and its implementations.
- `partition_collections.py` contains `BaseFramePartitionManager` interface and its implementations.
	- `BaseFramePartitionManager` manages 2D-array of `BaseFramePartition` object
- `axis_partition.py` contains `BaseFrameFullAxisPartition` and with the following hierarchy:
	```
	BaseFrameFullAxisPartition -> PandasOnRayFrameFullAxisPartition -> {PandasOnRayFrameFullColumnPartition, PandasOnRayFrameFullRowPartition}
	```
	- `BaseFrameFullAxisPartition` is a high level view onto BaseFramePartitionManager' data. It is more
	   convenient to operate on `BaseFrameFullAxisPartition` sometimes.
