## Implementation Note

### Object Hierarchy

- `remote_partition.py` contains `BaseRemotePartition` interface and its implementations.
- `partition_collections.py` contains `BaseBlockPartitions` interface and its implementations.
	- `BaseBlockPartitions` manages 2D-array of `BaseRemotePartition` object
- `axis_partition.py` contains `BaseAxisPartition` and with the following hierarchy:
	```
	BaseAxisPartition -> PandasOnRayAxisPartition -> {PandasOnRayColumnPartition, PandasOnRayRowPartition}
	```
	- `BaseAxisPartition` is a high level view onto BaseBlockPartitions' data. It is sometimes more
	   convenient to operate on `BaseAxisPartition'
