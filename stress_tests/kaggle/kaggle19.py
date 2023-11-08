#!/usr/bin/env python
# coding: utf-8
import matplotlib

matplotlib.use("PS")
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

warnings.filterwarnings("ignore")
train = pd.read_csv("train.csv")
train.info()
train.head()
print(
    "The average person kills {:.4f} players, 99% of people have {} kills or less, while the most kills ever recorded is {}.".format(
        train["kills"].mean(), train["kills"].quantile(0.99), train["kills"].max()
    )
)
data = train.copy()
data.loc[data["kills"] > data["kills"].quantile(0.99)] = "8+"
plt.figure(figsize=(15, 10))
sns.countplot(data["kills"].astype("str").sort_values())
plt.title("Kill Count", fontsize=15)
plt.show()
data = train.copy()
data = data[data["kills"] == 0]
plt.figure(figsize=(15, 10))
plt.title("Damage Dealt by 0 killers", fontsize=15)
plt.show()
print(
    "{} players ({:.4f}%) have won without a single kill!".format(
        len(data[data["winPlacePerc"] == 1]),
        100 * len(data[data["winPlacePerc"] == 1]) / len(train),
    )
)
data1 = train[train["damageDealt"] == 0].copy()
print(
    "{} players ({:.4f}%) have won without dealing damage!".format(
        len(data1[data1["winPlacePerc"] == 1]),
        100 * len(data1[data1["winPlacePerc"] == 1]) / len(train),
    )
)
kills = train.copy()
kills["killsCategories"] = pd.cut(
    kills["kills"],
    [-1, 0, 2, 5, 10, 60],
    labels=["0_kills", "1-2_kills", "3-5_kills", "6-10_kills", "10+_kills"],
)
plt.figure(figsize=(15, 8))
sns.boxplot(x="killsCategories", y="winPlacePerc", data=kills)
plt.show()
print(
    "The average person walks for {:.1f}m, 99% of people have walked {}m or less, while the marathoner champion walked for {}m.".format(
        train["walkDistance"].mean(),
        train["walkDistance"].quantile(0.99),
        train["walkDistance"].max(),
    )
)
data = train.copy()
data = data[data["walkDistance"] < train["walkDistance"].quantile(0.99)]
plt.figure(figsize=(15, 10))
plt.title("Walking Distance Distribution", fontsize=15)
sns.distplot(data["walkDistance"])
plt.show()
print(
    "{} players ({:.4f}%) walked 0 meters. This means that they die before even taking a step or they are afk (more possible).".format(
        len(data[data["walkDistance"] == 0]),
        100 * len(data1[data1["walkDistance"] == 0]) / len(train),
    )
)
print(
    "The average person drives for {:.1f}m, 99% of people have drived {}m or less, while the formula 1 champion drived for {}m.".format(
        train["rideDistance"].mean(),
        train["rideDistance"].quantile(0.99),
        train["rideDistance"].max(),
    )
)
data = train.copy()
data = data[data["rideDistance"] < train["rideDistance"].quantile(0.9)]
plt.figure(figsize=(15, 10))
plt.title("Ride Distance Distribution", fontsize=15)
sns.distplot(data["rideDistance"])
plt.show()
print(
    "{} players ({:.4f}%) drived for 0 meters. This means that they don't have a driving licence yet.".format(
        len(data[data["rideDistance"] == 0]),
        100 * len(data1[data1["rideDistance"] == 0]) / len(train),
    )
)
f, ax1 = plt.subplots(figsize=(20, 10))
sns.pointplot(
    x="vehicleDestroys", y="winPlacePerc", data=data, color="#606060", alpha=0.8
)
plt.xlabel("Number of Vehicle Destroys", fontsize=15, color="blue")
plt.ylabel("Win Percentage", fontsize=15, color="blue")
plt.title("Vehicle Destroys/ Win Ratio", fontsize=20, color="blue")
plt.grid()
plt.show()
print(
    "The average person swims for {:.1f}m, 99% of people have swimemd {}m or less, while the olympic champion swimmed for {}m.".format(
        train["swimDistance"].mean(),
        train["swimDistance"].quantile(0.99),
        train["swimDistance"].max(),
    )
)
data = train.copy()
data = data[data["swimDistance"] < train["swimDistance"].quantile(0.95)]
plt.figure(figsize=(15, 10))
plt.title("Swim Distance Distribution", fontsize=15)
sns.distplot(data["swimDistance"])
plt.show()
swim = train.copy()
swim["swimDistance"] = pd.cut(
    swim["swimDistance"], [-1, 0, 5, 20, 5286], labels=["0m", "1-5m", "6-20m", "20m+"]
)
plt.figure(figsize=(15, 8))
sns.boxplot(x="swimDistance", y="winPlacePerc", data=swim)
plt.show()
print(
    "The average person uses {:.1f} heal items, 99% of people use {} or less, while the doctor used {}.".format(
        train["heals"].mean(), train["heals"].quantile(0.99), train["heals"].max()
    )
)
print(
    "The average person uses {:.1f} boost items, 99% of people use {} or less, while the doctor used {}.".format(
        train["boosts"].mean(), train["boosts"].quantile(0.99), train["boosts"].max()
    )
)
data = train.copy()
data = data[data["heals"] < data["heals"].quantile(0.99)]
data = data[data["boosts"] < data["boosts"].quantile(0.99)]
f, ax1 = plt.subplots(figsize=(20, 10))
sns.pointplot(x="heals", y="winPlacePerc", data=data, color="lime", alpha=0.8)
sns.pointplot(x="boosts", y="winPlacePerc", data=data, color="blue", alpha=0.8)
plt.text(4, 0.6, "Heals", color="lime", fontsize=17, style="italic")
plt.text(4, 0.55, "Boosts", color="blue", fontsize=17, style="italic")
plt.xlabel("Number of heal/boost items", fontsize=15, color="blue")
plt.ylabel("Win Percentage", fontsize=15, color="blue")
plt.title("Heals vs Boosts", fontsize=20, color="blue")
plt.grid()
plt.show()
solos = train[train["numGroups"] > 50]
duos = train[(train["numGroups"] > 25) & (train["numGroups"] <= 50)]
squads = train[train["numGroups"] <= 25]
print(
    "There are {} ({:.2f}%) solo games, {} ({:.2f}%) duo games and {} ({:.2f}%) squad games.".format(
        len(solos),
        100 * len(solos) / len(train),
        len(duos),
        100 * len(duos) / len(train),
        len(squads),
        100 * len(squads) / len(train),
    )
)
f, ax1 = plt.subplots(figsize=(20, 10))
sns.pointplot(x="kills", y="winPlacePerc", data=solos, color="black", alpha=0.8)
sns.pointplot(x="kills", y="winPlacePerc", data=duos, color="#CC0000", alpha=0.8)
sns.pointplot(x="kills", y="winPlacePerc", data=squads, color="#3399FF", alpha=0.8)
plt.text(37, 0.6, "Solos", color="black", fontsize=17, style="italic")
plt.text(37, 0.55, "Duos", color="#CC0000", fontsize=17, style="italic")
plt.text(37, 0.5, "Squads", color="#3399FF", fontsize=17, style="italic")
plt.xlabel("Number of kills", fontsize=15, color="blue")
plt.ylabel("Win Percentage", fontsize=15, color="blue")
plt.title("Solo vs Duo vs Squad Kills", fontsize=20, color="blue")
plt.grid()
plt.show()
f, ax1 = plt.subplots(figsize=(20, 10))
sns.pointplot(x="DBNOs", y="winPlacePerc", data=duos, color="#CC0000", alpha=0.8)
sns.pointplot(x="DBNOs", y="winPlacePerc", data=squads, color="#3399FF", alpha=0.8)
sns.pointplot(x="assists", y="winPlacePerc", data=duos, color="#FF6666", alpha=0.8)
sns.pointplot(x="assists", y="winPlacePerc", data=squads, color="#CCE5FF", alpha=0.8)
sns.pointplot(x="revives", y="winPlacePerc", data=duos, color="#660000", alpha=0.8)
sns.pointplot(x="revives", y="winPlacePerc", data=squads, color="#000066", alpha=0.8)
plt.text(14, 0.5, "Duos - Assists", color="#FF6666", fontsize=17, style="italic")
plt.text(14, 0.45, "Duos - DBNOs", color="#CC0000", fontsize=17, style="italic")
plt.text(14, 0.4, "Duos - Revives", color="#660000", fontsize=17, style="italic")
plt.text(14, 0.35, "Squads - Assists", color="#CCE5FF", fontsize=17, style="italic")
plt.text(14, 0.3, "Squads - DBNOs", color="#3399FF", fontsize=17, style="italic")
plt.text(14, 0.25, "Squads - Revives", color="#000066", fontsize=17, style="italic")
plt.xlabel("Number of DBNOs/Assits/Revives", fontsize=15, color="blue")
plt.ylabel("Win Percentage", fontsize=15, color="blue")
plt.title("Duo vs Squad DBNOs, Assists, and Revives", fontsize=20, color="blue")
plt.grid()
plt.show()
f, ax = plt.subplots(figsize=(15, 15))
sns.heatmap(train.corr(), annot=True, linewidths=0.5, fmt=".1f", ax=ax)
plt.show()
k = 5  # number of variables for heatmap
f, ax = plt.subplots(figsize=(11, 11))
cols = train.corr().nlargest(k, "winPlacePerc")["winPlacePerc"].index
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(
    cm,
    cbar=True,
    annot=True,
    square=True,
    fmt=".2f",
    annot_kws={"size": 10},
    yticklabels=cols.values,
    xticklabels=cols.values,
)
plt.show()
train["playersJoined"] = train.groupby("matchId")["matchId"].transform("count")
data = train.copy()
data = data[data["playersJoined"] > 49]
train["killsNorm"] = train["kills"] * ((100 - train["playersJoined"]) / 100 + 1)
train["damageDealtNorm"] = train["damageDealt"] * (
    (100 - train["playersJoined"]) / 100 + 1
)
train[["playersJoined", "kills", "killsNorm", "damageDealt", "damageDealtNorm"]][5:8]
train["healsAndBoosts"] = train["heals"] + train["boosts"]
train["totalDistance"] = (
    train["walkDistance"] + train["rideDistance"] + train["swimDistance"]
)
train["boostsPerWalkDistance"] = train["boosts"] / (
    train["walkDistance"] + 1
)  # The +1 is to avoid infinity, because there are entries where boosts>0 and walkDistance=0. Strange.
train["boostsPerWalkDistance"].fillna(0, inplace=True)
train["healsPerWalkDistance"] = train["heals"] / (
    train["walkDistance"] + 1
)  # The +1 is to avoid infinity, because there are entries where heals>0 and walkDistance=0. Strange.
train["healsPerWalkDistance"].fillna(0, inplace=True)
train["healsAndBoostsPerWalkDistance"] = train["healsAndBoosts"] / (
    train["walkDistance"] + 1
)  # The +1 is to avoid infinity.
train["healsAndBoostsPerWalkDistance"].fillna(0, inplace=True)
train[
    [
        "walkDistance",
        "boosts",
        "boostsPerWalkDistance",
        "heals",
        "healsPerWalkDistance",
        "healsAndBoosts",
        "healsAndBoostsPerWalkDistance",
    ]
][40:45]
train["killsPerWalkDistance"] = train["kills"] / (
    train["walkDistance"] + 1
)  # The +1 is to avoid infinity, because there are entries where kills>0 and walkDistance=0. Strange.
train["killsPerWalkDistance"].fillna(0, inplace=True)
train[
    ["kills", "walkDistance", "rideDistance", "killsPerWalkDistance", "winPlacePerc"]
].sort_values(by="killsPerWalkDistance").tail(10)
train["team"] = [
    1 if i > 50 else 2 if (i > 25 & i <= 50) else 4 for i in train["numGroups"]
]
train.head()
