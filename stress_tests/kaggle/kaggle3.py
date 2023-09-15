#!/usr/bin/env python
import matplotlib

matplotlib.use("PS")
import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import seaborn as sns  # visualization tool

import modin.pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

data = pd.read_csv("pokemon.csv")
data.info()
data.corr()
f, ax = plt.subplots(figsize=(18, 18))
sns.heatmap(data.corr(), annot=True, linewidths=0.5, fmt=".1f", ax=ax)
data.head(10)
data.columns
data.Speed.plot(
    kind="line",
    color="g",
    label="Speed",
    linewidth=1,
    alpha=0.5,
    grid=True,
    linestyle=":",
)
data.Defense.plot(
    color="r", label="Defense", linewidth=1, alpha=0.5, grid=True, linestyle="-."
)
plt.legend(loc="upper right")  # legend = puts label into plot
plt.xlabel("x axis")  # label = name of label
plt.ylabel("y axis")
plt.title("Line Plot")  # title = title of plot
data.plot(kind="scatter", x="Attack", y="Defense", alpha=0.5, color="red")
plt.xlabel("Attack")  # label = name of label
plt.ylabel("Defence")
plt.title("Attack Defense Scatter Plot")  # title = title of plot
data.Speed.plot(kind="hist", bins=50, figsize=(12, 12))
data.Speed.plot(kind="hist", bins=50)
dictionary = {"spain": "madrid", "usa": "vegas"}
print(dictionary.keys())
print(dictionary.values())
dictionary["spain"] = "barcelona"  # update existing entry
print(dictionary)
dictionary["france"] = "paris"  # Add new entry
print(dictionary)
del dictionary["spain"]  # remove entry with key 'spain'
print(dictionary)
print("france" in dictionary)  # check include or not
dictionary.clear()  # remove all entries in dict
print(dictionary)
print(dictionary)  # it gives error because dictionary is deleted
data = pd.read_csv("pokemon.csv")
series = data["Defense"]  # data['Defense'] = series
print(type(series))
data_frame = data[["Defense"]]  # data[['Defense']] = data frame
print(type(data_frame))
print(3 > 2)
print(3 != 2)
print(True and False)
print(True or False)
x = (
    data["Defense"] > 200
)  # There are only 3 pokemons who have higher defense value than 200
data[x]
data[np.logical_and(data["Defense"] > 200, data["Attack"] > 100)]
data[(data["Defense"] > 200) & (data["Attack"] > 100)]
i = 0
while i != 5:
    print("i is: ", i)
    i += 1
print(i, " is equal to 5")
lis = [1, 2, 3, 4, 5]
for i in lis:
    print("i is: ", i)
print("")
for index, value in enumerate(lis):
    print(index, " : ", value)
print("")
dictionary = {"spain": "madrid", "france": "paris"}
for key, value in dictionary.items():
    print(key, " : ", value)
print("")
for index, value in data[["Attack"]][0:1].iterrows():
    print(index, " : ", value)


def tuble_ex():
    """return defined t tuble"""
    t = (1, 2, 3)
    return t


a, b, c = tuble_ex()
print(a, b, c)
x = 2


def f():
    x = 3
    return x


print(x)  # x = 2 global scope
print(f())  # x = 3 local scope
x = 5


def f():
    y = 2 * x  # there is no local scope x
    return y


print(f())  # it uses global scope x
import builtins

dir(builtins)


def square():
    """return square of value"""

    def add():
        """add two local variable"""
        x = 2
        y = 3
        z = x + y
        return z

    return add() ** 2


print(square())


def f(a, b=1, c=2):
    y = a + b + c
    return y


print(f(5))
print(f(5, 4, 3))


def f(*args):
    for i in args:
        print(i)


f(1)
print("")
f(1, 2, 3, 4)


def f(**kwargs):
    """print key and value of dictionary"""
    for (
        key,
        value,
    ) in (
        kwargs.items()
    ):  # If you do not understand this part turn for loop part and look at dictionary in for loop
        print(key, " ", value)


f(country="spain", capital="madrid", population=123456)
number_list = [1, 2, 3]
y = map(lambda x: x**2, number_list)
print(list(y))
name = "ronaldo"
it = iter(name)
print(next(it))  # print next iteration
print(*it)  # print remaining iteration
list1 = [1, 2, 3, 4]
list2 = [5, 6, 7, 8]
z = zip(list1, list2)
print(z)
z_list = list(z)
print(z_list)
un_zip = zip(*z_list)
un_list1, un_list2 = list(un_zip)  # unzip returns tuble
print(un_list1)
print(un_list2)
print(type(un_list2))
num1 = [1, 2, 3]
num2 = [i + 1 for i in num1]
print(num2)
num1 = [5, 10, 15]
num2 = [i**2 if i == 10 else i - 5 if i < 7 else i + 5 for i in num1]
print(num2)
threshold = sum(data.Speed) / len(data.Speed)
data["speed_level"] = ["high" if i > threshold else "low" for i in data.Speed]
data.loc[:10, ["speed_level", "Speed"]]  # we will learn loc more detailed later
data = pd.read_csv("pokemon.csv")
data.head()  # head shows first 5 rows
data.tail()
data.columns
data.shape
data.info()
print(
    data["Type 1"].value_counts(dropna=False)
)  # if there are nan values that also be counted
data.describe()  # ignore null entries
data.boxplot(column="Attack", by="Legendary")
data_new = data.head()  # I only take 5 rows into new data
data_new
melted = pd.melt(frame=data_new, id_vars="Name", value_vars=["Attack", "Defense"])
melted
melted.pivot(index="Name", columns="variable", values="value")
data1 = data.head()
data2 = data.tail()
conc_data_row = pd.concat(
    [data1, data2], axis=0, ignore_index=True
)  # axis = 0 : adds dataframes in row
conc_data_row
data1 = data["Attack"].head()
data2 = data["Defense"].head()
conc_data_col = pd.concat([data1, data2], axis=1)  # axis = 0 : adds dataframes in row
conc_data_col
data.dtypes
data["Type 1"] = data["Type 1"].astype("category")
data["Speed"] = data["Speed"].astype("float")
data.dtypes
data.info()
data["Type 2"].value_counts(dropna=False)
data1 = (
    data  # also we will use data to fill missing value so I assign it to data1 variable
)
data1["Type 2"].dropna(
    inplace=True
)  # inplace = True means we do not assign it to new variable. Changes automatically assigned to data
assert 1 == 1  # return nothing because it is true
assert data["Type 2"].notnull().all()  # returns nothing because we drop nan values
data["Type 2"].fillna("empty", inplace=True)
assert (
    data["Type 2"].notnull().all()
)  # returns nothing because we do not have nan values
country = ["Spain", "France"]
population = ["11", "12"]
list_label = ["country", "population"]
list_col = [country, population]
zipped = list(zip(list_label, list_col))
data_dict = dict(zipped)
df = pd.DataFrame(data_dict)
df
df["capital"] = ["madrid", "paris"]
df
df["income"] = 0  # Broadcasting entire column
df
data1 = data.loc[:, ["Attack", "Defense", "Speed"]]
data1.plot()
data1.plot(subplots=True)
plt.show()
data1.plot(kind="scatter", x="Attack", y="Defense")
plt.show()
data1.plot(kind="hist", y="Defense", bins=50, range=(0, 250), normed=True)
fig, axes = plt.subplots(nrows=2, ncols=1)
data1.plot(kind="hist", y="Defense", bins=50, range=(0, 250), normed=True, ax=axes[0])
data1.plot(
    kind="hist",
    y="Defense",
    bins=50,
    range=(0, 250),
    normed=True,
    ax=axes[1],
    cumulative=True,
)
plt.savefig("graph.png")
plt
data.describe()
time_list = ["1992-03-08", "1992-04-12"]
print(type(time_list[1]))  # As you can see date is string
datetime_object = pd.to_datetime(time_list)
print(type(datetime_object))
import warnings

warnings.filterwarnings("ignore")
data2 = data.head()
date_list = ["1992-01-10", "1992-02-10", "1992-03-10", "1993-03-15", "1993-03-16"]
datetime_object = pd.to_datetime(date_list)
data2["date"] = datetime_object
data2 = data2.set_index("date")
data2
print(data2.loc["1993-03-16"])
print(data2.loc["1992-03-10":"1993-03-16"])
data2.resample("A").mean()
data2.resample("M").mean()
data2.resample("M").first().interpolate("linear")
data2.resample("M").mean().interpolate("linear")
data = pd.read_csv("pokemon.csv")
data = data.set_index("#")
data.head()
data["HP"][1]
data.HP[1]
data.loc[1, ["HP"]]
data[["HP", "Attack"]]
print(type(data["HP"]))  # series
print(type(data[["HP"]]))  # data frames
data.loc[1:10, "HP":"Defense"]  # 10 and "Defense" are inclusive
data.loc[10:1:-1, "HP":"Defense"]
data.loc[1:10, "Speed":]
boolean = data.HP > 200
data[boolean]
first_filter = data.HP > 150
second_filter = data.Speed > 35
data[first_filter & second_filter]
data.HP[data.Speed < 15]


def div(n):
    return n / 2


data.HP.apply(div)
data.HP.apply(lambda n: n / 2)
data["total_power"] = data.Attack + data.Defense
data.head()
print(data.index.name)
data.index.name = "index_name"
data.head()
data.head()
data3 = data.copy()
data3.index = range(100, 100 + len(data3.index), 1)
data3.head()
data = pd.read_csv("pokemon.csv")
data.head()
data1 = data.set_index(["Type 1", "Type 2"])
data1.head(100)
dic = {
    "treatment": ["A", "A", "B", "B"],
    "gender": ["F", "M", "F", "M"],
    "response": [10, 45, 5, 9],
    "age": [15, 4, 72, 65],
}
df = pd.DataFrame(dic)
df
df.pivot(index="treatment", columns="gender", values="response")
df1 = df.set_index(["treatment", "gender"])
df1
df1.unstack(level=0)
df1.unstack(level=1)
df2 = df1.swaplevel(0, 1)
df2
df
pd.melt(df, id_vars="treatment", value_vars=["age", "response"])
df
df.groupby("treatment").mean()  # mean is aggregation / reduce method
df.groupby("treatment").age.max()
df.groupby("treatment")[["age", "response"]].min()
df.info()
