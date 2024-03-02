import matplotlib

matplotlib.use("PS")
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
    VotingClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

import modin.pandas as pd

sns.set(style="white", context="notebook", palette="deep")
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
IDtest = test["PassengerId"]


def detect_outliers(df, n, features):
    outlier_indices = []
    for col in features:
        Q1 = np.percentile(df[col], 25)
        Q3 = np.percentile(df[col], 75)
        IQR = Q3 - Q1
        outlier_step = 1.5 * IQR
        outlier_list_col = df[
            (df[col] < Q1 - outlier_step) | (df[col] > Q3 + outlier_step)
        ].index
        outlier_indices.extend(outlier_list_col)
    outlier_indices = Counter(outlier_indices)
    multiple_outliers = [k for k, v in outlier_indices.items() if v > n]
    return multiple_outliers


Outliers_to_drop = detect_outliers(train, 2, ["Age", "SibSp", "Parch", "Fare"])
train.loc[Outliers_to_drop]  # Show the outliers rows
train = train.drop(Outliers_to_drop, axis=0).reset_index(drop=True)
train_len = len(train)
dataset = pd.concat(list_of_objs=[train, test], axis=0).reset_index(drop=True)
dataset = dataset.fillna(np.nan)
dataset.isnull().sum()
train.info()
train.isnull().sum()
train.head()
train.dtypes
train.describe()
g = sns.heatmap(
    train[["Survived", "SibSp", "Parch", "Age", "Fare"]].corr(),
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
)
g = sns.factorplot(
    x="SibSp", y="Survived", data=train, kind="bar", size=6, palette="muted"
)
g.despine(left=True)
g = g.set_ylabels("survival probability")
g = sns.factorplot(
    x="Parch", y="Survived", data=train, kind="bar", size=6, palette="muted"
)
g.despine(left=True)
g = g.set_ylabels("survival probability")
dataset["Fare"].isnull().sum()
dataset["Fare"] = dataset["Fare"].fillna(dataset["Fare"].median())
g = sns.distplot(
    dataset["Fare"], color="m", label="Skewness : %.2f" % (dataset["Fare"].skew())
)
g = g.legend(loc="best")
dataset["Fare"] = dataset["Fare"].map(lambda i: np.log(i) if i > 0 else 0)
g = sns.distplot(
    dataset["Fare"], color="b", label="Skewness : %.2f" % (dataset["Fare"].skew())
)
g = g.legend(loc="best")
g = sns.barplot(x="Sex", y="Survived", data=train)
g = g.set_ylabel("Survival Probability")
train[["Sex", "Survived"]].groupby("Sex").mean()
g = sns.factorplot(
    x="Pclass", y="Survived", data=train, kind="bar", size=6, palette="muted"
)
g.despine(left=True)
g = g.set_ylabels("survival probability")
g = sns.factorplot(
    x="Pclass", y="Survived", hue="Sex", data=train, size=6, kind="bar", palette="muted"
)
g.despine(left=True)
g = g.set_ylabels("survival probability")
dataset["Embarked"].isnull().sum()
dataset["Embarked"] = dataset["Embarked"].fillna("S")
g = sns.factorplot(
    x="Embarked", y="Survived", data=train, size=6, kind="bar", palette="muted"
)
g.despine(left=True)
g = g.set_ylabels("survival probability")
g = sns.factorplot(
    "Pclass", col="Embarked", data=train, size=6, kind="count", palette="muted"
)
g.despine(left=True)
g = g.set_ylabels("Count")
g = sns.factorplot(y="Age", x="Sex", data=dataset, kind="box")
g = sns.factorplot(y="Age", x="Sex", hue="Pclass", data=dataset, kind="box")
g = sns.factorplot(y="Age", x="Parch", data=dataset, kind="box")
g = sns.factorplot(y="Age", x="SibSp", data=dataset, kind="box")
dataset["Sex"] = dataset["Sex"].map({"male": 0, "female": 1})
g = sns.heatmap(
    dataset[["Age", "Sex", "SibSp", "Parch", "Pclass"]].corr(), cmap="BrBG", annot=True
)
index_NaN_age = list(dataset["Age"][dataset["Age"].isnull()].index)
for i in index_NaN_age:
    age_med = dataset["Age"].median()
    age_pred = dataset["Age"][
        (
            (dataset["SibSp"] == dataset.iloc[i]["SibSp"])
            & (dataset["Parch"] == dataset.iloc[i]["Parch"])
            & (dataset["Pclass"] == dataset.iloc[i]["Pclass"])
        )
    ].median()
    if not np.isnan(age_pred):
        dataset["Age"].iloc[i] = age_pred
    else:
        dataset["Age"].iloc[i] = age_med
g = sns.factorplot(x="Survived", y="Age", data=train, kind="box")
g = sns.factorplot(x="Survived", y="Age", data=train, kind="violin")
dataset["Name"].head()
dataset_title = [i.split(",")[1].split(".")[0].strip() for i in dataset["Name"]]
dataset["Title"] = pd.Series(dataset_title)
dataset["Title"].head()
g = sns.countplot(x="Title", data=dataset)
g = plt.setp(g.get_xticklabels(), rotation=45)
dataset["Title"] = dataset["Title"].replace(
    [
        "Lady",
        "the Countess",
        "Countess",
        "Capt",
        "Col",
        "Don",
        "Dr",
        "Major",
        "Rev",
        "Sir",
        "Jonkheer",
        "Dona",
    ],
    "Rare",
)
dataset["Title"] = dataset["Title"].map(
    {"Master": 0, "Miss": 1, "Ms": 1, "Mme": 1, "Mlle": 1, "Mrs": 1, "Mr": 2, "Rare": 3}
)
dataset["Title"] = dataset["Title"].astype(int)
g = sns.countplot(dataset["Title"])
g = g.set_xticklabels(["Master", "Miss/Ms/Mme/Mlle/Mrs", "Mr", "Rare"])
g = sns.factorplot(x="Title", y="Survived", data=dataset, kind="bar")
g = g.set_xticklabels(["Master", "Miss-Mrs", "Mr", "Rare"])
g = g.set_ylabels("survival probability")
dataset.drop(labels=["Name"], axis=1, inplace=True)
dataset["Fsize"] = dataset["SibSp"] + dataset["Parch"] + 1
g = sns.factorplot(x="Fsize", y="Survived", data=dataset)
g = g.set_ylabels("Survival Probability")
dataset["Single"] = dataset["Fsize"].map(lambda s: 1 if s == 1 else 0)
dataset["SmallF"] = dataset["Fsize"].map(lambda s: 1 if s == 2 else 0)
dataset["MedF"] = dataset["Fsize"].map(lambda s: 1 if 3 <= s <= 4 else 0)
dataset["LargeF"] = dataset["Fsize"].map(lambda s: 1 if s >= 5 else 0)
g = sns.factorplot(x="Single", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="SmallF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="MedF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
g = sns.factorplot(x="LargeF", y="Survived", data=dataset, kind="bar")
g = g.set_ylabels("Survival Probability")
dataset = pd.get_dummies(dataset, columns=["Title"])
dataset = pd.get_dummies(dataset, columns=["Embarked"], prefix="Em")
dataset.head()
dataset["Cabin"].head()
dataset["Cabin"].describe()
dataset["Cabin"].isnull().sum()
dataset["Cabin"][dataset["Cabin"].notnull()].head()
dataset["Cabin"] = pd.Series(
    [i[0] if not pd.isnull(i) else "X" for i in dataset["Cabin"]]
)
g = sns.countplot(dataset["Cabin"], order=["A", "B", "C", "D", "E", "F", "G", "T", "X"])
g = sns.factorplot(
    y="Survived",
    x="Cabin",
    data=dataset,
    kind="bar",
    order=["A", "B", "C", "D", "E", "F", "G", "T", "X"],
)
g = g.set_ylabels("Survival Probability")
dataset = pd.get_dummies(dataset, columns=["Cabin"], prefix="Cabin")
dataset["Ticket"].head()
Ticket = []
for i in list(dataset.Ticket):
    if not i.isdigit():
        Ticket.append(
            i.replace(".", "").replace("/", "").strip().split(" ")[0]
        )  # Take prefix
    else:
        Ticket.append("X")
dataset["Ticket"] = Ticket
dataset["Ticket"].head()
dataset = pd.get_dummies(dataset, columns=["Ticket"], prefix="T")
dataset["Pclass"] = dataset["Pclass"].astype("category")
dataset = pd.get_dummies(dataset, columns=["Pclass"], prefix="Pc")
dataset.drop(labels=["PassengerId"], axis=1, inplace=True)
dataset.head()
train = dataset[:train_len]
test = dataset[train_len:]
test.drop(labels=["Survived"], axis=1, inplace=True)
train["Survived"] = train["Survived"].astype(int)
Y_train = train["Survived"]
X_train = train.drop(labels=["Survived"], axis=1)
kfold = StratifiedKFold(n_splits=10)
random_state = 2
classifiers = []
classifiers.append(SVC(random_state=random_state))
classifiers.append(DecisionTreeClassifier(random_state=random_state))
classifiers.append(
    AdaBoostClassifier(
        DecisionTreeClassifier(random_state=random_state),
        random_state=random_state,
        learning_rate=0.1,
    )
)
classifiers.append(RandomForestClassifier(random_state=random_state))
classifiers.append(ExtraTreesClassifier(random_state=random_state))
classifiers.append(GradientBoostingClassifier(random_state=random_state))
classifiers.append(MLPClassifier(random_state=random_state))
classifiers.append(KNeighborsClassifier())
classifiers.append(LogisticRegression(random_state=random_state))
classifiers.append(LinearDiscriminantAnalysis())
cv_results = []
for classifier in classifiers:
    cv_results.append(
        cross_val_score(
            classifier, X_train, y=Y_train, scoring="accuracy", cv=kfold, n_jobs=4
        )
    )
cv_means = []
cv_std = []
for cv_result in cv_results:
    cv_means.append(cv_result.mean())
    cv_std.append(cv_result.std())
cv_res = pd.DataFrame(
    {
        "CrossValMeans": cv_means,
        "CrossValerrors": cv_std,
        "Algorithm": [
            "SVC",
            "DecisionTree",
            "AdaBoost",
            "RandomForest",
            "ExtraTrees",
            "GradientBoosting",
            "MultipleLayerPerceptron",
            "KNeighboors",
            "LogisticRegression",
            "LinearDiscriminantAnalysis",
        ],
    }
)
g = sns.barplot(
    "CrossValMeans",
    "Algorithm",
    data=cv_res,
    palette="Set3",
    orient="h",
    **{"xerr": cv_std}
)
g.set_xlabel("Mean Accuracy")
g = g.set_title("Cross validation scores")
DTC = DecisionTreeClassifier()
adaDTC = AdaBoostClassifier(DTC, random_state=7)
ada_param_grid = {
    "base_estimator__criterion": ["gini", "entropy"],
    "base_estimator__splitter": ["best", "random"],
    "algorithm": ["SAMME", "SAMME.R"],
    "n_estimators": [1, 2],
    "learning_rate": [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 1.5],
}
gsadaDTC = GridSearchCV(
    adaDTC, param_grid=ada_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1
)
gsadaDTC.fit(X_train, Y_train)
ada_best = gsadaDTC.best_estimator_
gsadaDTC.best_score_
ExtC = ExtraTreesClassifier()
ex_param_grid = {
    "max_depth": [None],
    "max_features": [1, 3, 10],
    "min_samples_split": [2, 3, 10],
    "min_samples_leaf": [1, 3, 10],
    "bootstrap": [False],
    "n_estimators": [100, 300],
    "criterion": ["gini"],
}
gsExtC = GridSearchCV(
    ExtC, param_grid=ex_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1
)
gsExtC.fit(X_train, Y_train)
ExtC_best = gsExtC.best_estimator_
gsExtC.best_score_
RFC = RandomForestClassifier()
rf_param_grid = {
    "max_depth": [None],
    "max_features": [1, 3, 10],
    "min_samples_split": [2, 3, 10],
    "min_samples_leaf": [1, 3, 10],
    "bootstrap": [False],
    "n_estimators": [100, 300],
    "criterion": ["gini"],
}
gsRFC = GridSearchCV(
    RFC, param_grid=rf_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1
)
gsRFC.fit(X_train, Y_train)
RFC_best = gsRFC.best_estimator_
gsRFC.best_score_
GBC = GradientBoostingClassifier()
gb_param_grid = {
    "loss": ["deviance"],
    "n_estimators": [100, 200, 300],
    "learning_rate": [0.1, 0.05, 0.01],
    "max_depth": [4, 8],
    "min_samples_leaf": [100, 150],
    "max_features": [0.3, 0.1],
}
gsGBC = GridSearchCV(
    GBC, param_grid=gb_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1
)
gsGBC.fit(X_train, Y_train)
GBC_best = gsGBC.best_estimator_
gsGBC.best_score_
SVMC = SVC(probability=True)
svc_param_grid = {
    "kernel": ["rbf"],
    "gamma": [0.001, 0.01, 0.1, 1],
    "C": [1, 10, 50, 100, 200, 300, 1000],
}
gsSVMC = GridSearchCV(
    SVMC, param_grid=svc_param_grid, cv=kfold, scoring="accuracy", n_jobs=4, verbose=1
)
gsSVMC.fit(X_train, Y_train)
SVMC_best = gsSVMC.best_estimator_
gsSVMC.best_score_


def plot_learning_curve(
    estimator,
    title,
    X,
    y,
    ylim=None,
    cv=None,
    n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5),
):
    """Generate a simple plot of the test and training learning curve"""
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes
    )
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()
    plt.fill_between(
        train_sizes,
        train_scores_mean - train_scores_std,
        train_scores_mean + train_scores_std,
        alpha=0.1,
        color="r",
    )
    plt.fill_between(
        train_sizes,
        test_scores_mean - test_scores_std,
        test_scores_mean + test_scores_std,
        alpha=0.1,
        color="g",
    )
    plt.plot(train_sizes, train_scores_mean, "o-", color="r", label="Training score")
    plt.plot(
        train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
    )
    plt.legend(loc="best")
    return plt


g = plot_learning_curve(
    gsRFC.best_estimator_, "RF mearning curves", X_train, Y_train, cv=kfold
)
g = plot_learning_curve(
    gsExtC.best_estimator_, "ExtraTrees learning curves", X_train, Y_train, cv=kfold
)
g = plot_learning_curve(
    gsSVMC.best_estimator_, "SVC learning curves", X_train, Y_train, cv=kfold
)
g = plot_learning_curve(
    gsadaDTC.best_estimator_, "AdaBoost learning curves", X_train, Y_train, cv=kfold
)
g = plot_learning_curve(
    gsGBC.best_estimator_,
    "GradientBoosting learning curves",
    X_train,
    Y_train,
    cv=kfold,
)
nrows = ncols = 2
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharex="all", figsize=(15, 15))
names_classifiers = [
    ("AdaBoosting", ada_best),
    ("ExtraTrees", ExtC_best),
    ("RandomForest", RFC_best),
    ("GradientBoosting", GBC_best),
]
nclassifier = 0
for row in range(nrows):
    for col in range(ncols):
        name = names_classifiers[nclassifier][0]
        classifier = names_classifiers[nclassifier][1]
        indices = np.argsort(classifier.feature_importances_)[::-1][:40]
        g = sns.barplot(
            y=X_train.columns[indices][:40],
            x=classifier.feature_importances_[indices][:40],
            orient="h",
            ax=axes[row][col],
        )
        g.set_xlabel("Relative importance", fontsize=12)
        g.set_ylabel("Features", fontsize=12)
        g.tick_params(labelsize=9)
        g.set_title(name + " feature importance")
        nclassifier += 1
test_Survived_RFC = pd.Series(RFC_best.predict(test), name="RFC")
test_Survived_ExtC = pd.Series(ExtC_best.predict(test), name="ExtC")
test_Survived_SVMC = pd.Series(SVMC_best.predict(test), name="SVC")
test_Survived_AdaC = pd.Series(ada_best.predict(test), name="Ada")
test_Survived_GBC = pd.Series(GBC_best.predict(test), name="GBC")
ensemble_results = pd.concat(
    [
        test_Survived_RFC,
        test_Survived_ExtC,
        test_Survived_AdaC,
        test_Survived_GBC,
        test_Survived_SVMC,
    ],
    axis=1,
)
g = sns.heatmap(ensemble_results.corr(), annot=True)
votingC = VotingClassifier(
    estimators=[
        ("rfc", RFC_best),
        ("extc", ExtC_best),
        ("svc", SVMC_best),
        ("adac", ada_best),
        ("gbc", GBC_best),
    ],
    voting="soft",
    n_jobs=4,
)
votingC = votingC.fit(X_train, Y_train)
test_Survived = pd.Series(votingC.predict(test), name="Survived")
results = pd.concat([IDtest, test_Survived], axis=1)
results.to_csv("ensemble_python_voting.csv", index=False)
