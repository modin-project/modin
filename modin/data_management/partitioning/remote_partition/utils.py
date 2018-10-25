import pandas


def length_fn_pandas(df):
    assert isinstance(df, (pandas.DataFrame, pandas.Series))
    return len(df)


def width_fn_pandas(df):
    assert isinstance(df, (pandas.DataFrame, pandas.Series))
    if isinstance(df, pandas.DataFrame):
        return len(df.columns)
    else:
        return 1
