import pandas
from modin.engines.ray.pandas_on_ray.frame.partition import PandasOnRayFramePartition
from modin import __execution_engine__

if __execution_engine__ == "Ray":
    import ray
    import pyarrow


class PyarrowOnRayFramePartition(PandasOnRayFramePartition):
    def to_pandas(self):
        """Convert the object stored in this partition to a Pandas DataFrame.

        Returns:
            A Pandas DataFrame.
        """
        dataframe = self.get().to_pandas()
        assert type(dataframe) is pandas.DataFrame or type(dataframe) is pandas.Series

        return dataframe

    @classmethod
    def put(cls, obj):
        """Put an object in the Plasma store and wrap it in this object.

        Args:
            obj: The object to be put.

        Returns:
            A `RayRemotePartition` object.
        """
        return PyarrowOnRayFramePartition(ray.put(pyarrow.Table.from_pandas(obj)))

    @classmethod
    def length_extraction_fn(cls):
        return lambda table: table.num_rows

    @classmethod
    def width_extraction_fn(cls):
        return lambda table: table.num_columns - (1 if "index" in table.columns else 0)

    @classmethod
    def empty(cls):
        return cls.put(pandas.DataFrame())
