## Implementation Note

### Object Hierarchy

- `remote_partition.py` contains `RemotePartition` interface and its implementations.
- `partition_collections.py` contains `BlockPartitions` interface and its implementations.
	- `BlockPartitions` manages 2D-array of `RemotePartition` object
- `axis_partition.py` contains `AxisPartition` and with the following hierarchy:
	```
	AxisPartition -> RayAxisPartition -> {RayColumnPartition, RayRowPartition}
	```
	- `AxisPartition` is a high level view onto BlockPartitions' data. It is more
	   convient to operate on `AxisPartition` sometimes.