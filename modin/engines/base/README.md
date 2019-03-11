## Implementation Note

### Object Hierarchy

- `frame/partition.py` contains `BaseFramePartition` interface and its implementations.
- `frame/partition_manager.py` contains `BaseFrameManager` interface and its implementations.
	- `BaseFrameManager` manages 2D-array of `BaseFramePartition` object
- `frame/axis_partition.py` contains `BaseFrameAxisPartition` and with the following hierarchy:
	```
	BaseFrameAxisPartition -> PandasOnRayFrameAxisPartition -> {PandasOnRayFrameColumnPartition, PandasOnRayFrameRowPartition}
	```
	- `BaseFrameAxisPartition` is a high level view onto BaseFrameManager' data. It is more
	   convenient to operate on `BaseFrameAxisPartition` sometimes.
