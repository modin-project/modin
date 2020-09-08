import modin
import modin.data_management.functions


if __name__ == "__main__":
    import modin.pandas as pd

    print(pd.__version__)
    print(pd.DataFrame([1,2,3]))
