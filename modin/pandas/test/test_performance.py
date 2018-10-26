import numpy as np
import modin.pandas as pd

frame_data = {
    "col1": [0, 1, 2, 3],
    "col2": [4, 5, 6, 7],
    "col3": [8, 9, 0, 1],
    "col4": [2, 4, 5, 6],
}

nan_data = {
    "col1": [np.nan, 1, 2, 3],
    "col2": [4, 5, 6, np.nan],
    "col3": [8, 9, np.nan, 1],
    "col4": [2, np.nan, 5, np.nan],
}

frame_data2 = {"col5": [0], "col6": [1]}

# ray_df = pd.read_csv("PATH-TO-2017.csv")
# drop_cols = ["Unnamed: " + str(n) for n in [0,4,5,7]]
# ray_df = ray_df.drop(columns=drop_cols)
ray_df = pd.DataFrame(frame_data)

# def test_read_csv(benchmark):
#     path = "~/Downloads/201710k.csv"
#     result = benchmark(pd.read_csv, path)
#     print(result)


def test_sum(benchmark):
    # ray_df = pd.DataFrame(frame_data)
    result = benchmark(pd.DataFrame.sum, ray_df)
    return result


def test_fillna(benchmark):
    result = benchmark(pd.DataFrame.fillna, ray_df, 0)
    return result


def test_add(benchmark):
    result = benchmark(pd.DataFrame.add, ray_df, 1)
    return result


def test_add_df(benchmark):
    result = benchmark(pd.DataFrame.add, ray_df, ray_df)
    return result


def test_describe(benchmark):
    result = benchmark(pd.DataFrame.describe, ray_df)
    return result


def test_isna(benchmark):
    result = benchmark(pd.DataFrame.isna, ray_df)
    return result


# groupby
# join
# concat
# nlargest
