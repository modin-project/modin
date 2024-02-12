class TestModin:
    def test_modin(self) -> None:
        """
        Test some basic methods of the dataframe consortium standard.

        Full testing is done at https://github.com/data-apis/dataframe-api-compat,
        this is just to check that the entry point works as expected.
        """
        import modin.pandas as pd

        df_pd = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        df = df_pd.__dataframe_consortium_standard__()
        result_1 = df.get_column_names()
        expected_1 = ["a", "b"]
        assert result_1 == expected_1

        ser = pd.Series([1, 2, 3], name="a")
        assert ser.name == "a"
