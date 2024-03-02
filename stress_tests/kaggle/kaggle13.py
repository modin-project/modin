#!/usr/bin/env python
import matplotlib

matplotlib.use("PS")
import warnings  # current version of seaborn generates a bunch of warnings that we'll ignore

import modin.pandas as pd

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="white", color_codes=True)
iris = pd.read_csv("Iris.csv")  # the iris dataset is now a Pandas DataFrame
iris.head()
iris["Species"].value_counts()
iris.plot(kind="scatter", x="SepalLengthCm", y="SepalWidthCm")
sns.jointplot(x="SepalLengthCm", y="SepalWidthCm", data=iris, size=5)
sns.FacetGrid(iris, hue="Species", size=5).map(
    plt.scatter, "SepalLengthCm", "SepalWidthCm"
).add_legend()
sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.boxplot(x="Species", y="PetalLengthCm", data=iris)
ax = sns.stripplot(
    x="Species", y="PetalLengthCm", data=iris, jitter=True, edgecolor="gray"
)
sns.violinplot(x="Species", y="PetalLengthCm", data=iris, size=6)
sns.FacetGrid(iris, hue="Species", size=6).map(
    sns.kdeplot, "PetalLengthCm"
).add_legend()
iris.drop("Id", axis=1).boxplot(by="Species", figsize=(12, 6))
from pandas.tools.plotting import andrews_curves

andrews_curves(iris.drop("Id", axis=1), "Species")
from pandas.tools.plotting import parallel_coordinates

parallel_coordinates(iris.drop("Id", axis=1), "Species")
from pandas.tools.plotting import radviz

radviz(iris.drop("Id", axis=1), "Species")
