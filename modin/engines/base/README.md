## Implementation Note

### Object Hierarchy

- `frame/partition.py` contains `BasePandasFramePartition` interface and its implementations.
- `frame/partition_manager.py` contains `BasePandasFrameManager` interface and its implementations.
	- `BasePandasFrameManager` manages 2D-array of `BasePandasFramePartition` object
- `frame/axis_partition.py` contains `BaseFrameAxisPartition` and with the following hierarchy:
	```
	BaseFrameAxisPartition -> PandasOnRayFrameAxisPartition -> {PandasOnRayFrameColumnPartition, PandasOnRayFrameRowPartition}
	```
	- `BaseFrameAxisPartition` is a high level view onto BasePandasFrameManager' data. It is more
	   convenient to operate on `BaseFrameAxisPartition` sometimes.
