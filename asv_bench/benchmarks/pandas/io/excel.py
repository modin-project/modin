from io import BytesIO

import numpy as np

from modin.pandas import (
    DataFrame,
    ExcelWriter,
    date_range,
    read_excel,
)

import pandas._testing as tm


def _generate_dataframe():
    N = 2000
    C = 5
    df = DataFrame(
        np.random.randn(N, C),
        columns=[f"float{i}" for i in range(C)],
        index=date_range("20000101", periods=N, freq="H"),
    )
    df["object"] = tm.makeStringIndex(N)
    return df


class WriteExcel:

    params = ["openpyxl", "xlsxwriter", "xlwt"]
    param_names = ["engine"]

    def setup(self, engine):
        self.df = _generate_dataframe()

    def time_write_excel(self, engine):
        bio = BytesIO()
        bio.seek(0)
        writer = ExcelWriter(bio, engine=engine)
        self.df.to_excel(writer, sheet_name="Sheet1")
        writer.save()


class WriteExcelStyled:
    params = ["openpyxl", "xlsxwriter"]
    param_names = ["engine"]

    def setup(self, engine):
        self.df = _generate_dataframe()

    def time_write_excel_style(self, engine):
        bio = BytesIO()
        bio.seek(0)
        writer = ExcelWriter(bio, engine=engine)
        df_style = self.df.style
        df_style.applymap(lambda x: "border: red 1px solid;")
        df_style.applymap(lambda x: "color: blue")
        df_style.applymap(lambda x: "border-color: green black", subset=["float1"])
        df_style.to_excel(writer, sheet_name="Sheet1")
        writer.save()


from ..pandas_vb_common import setup  # noqa: F401 isort:skip
