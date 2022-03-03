from packaging import version
import pandas


pandas_version = pandas.__version__
if version.parse("1.1.0") <= version.parse(pandas_version) <= version.parse("1.1.5"):
    from .py36.io import (
        read_csv,
        read_json,
        read_table,
        read_parquet,
        read_gbq,
        read_excel,
        read_pickle,
        read_stata,
        read_feather,
        to_pickle,
    )

    __all__ = [
        "read_csv",
        "read_json",
        "read_table",
        "read_parquet",
        "read_gbq",
        "read_excel",
        "read_pickle",
        "read_stata",
        "read_feather",
        "to_pickle",
    ]
elif version.parse("1.4.0") <= version.parse(pandas_version) <= version.parse("1.4.99"):
    from .latest.io import (
        read_xml,
        read_csv,
        read_json,
        read_table,
        read_parquet,
        read_gbq,
        read_excel,
        read_pickle,
        read_stata,
        read_feather,
        to_pickle,
    )
    from pandas import Flags, Float32Dtype, Float64Dtype

    __all__ = [
        "read_xml",
        "read_csv",
        "read_json",
        "read_table",
        "read_parquet",
        "read_gbq",
        "read_excel",
        "read_pickle",
        "read_stata",
        "read_feather",
        "to_pickle",
        "Flags",
        "Float32Dtype",
        "Float64Dtype",
    ]
