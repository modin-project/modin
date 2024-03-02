import matplotlib

matplotlib.use("PS")
import warnings

import numpy as np
from sklearn.preprocessing import LabelEncoder

import modin.pandas as pd

warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import seaborn as sns

app_train = pd.read_csv("application_train.csv")
print("Training data shape: ", app_train.shape)
app_train.head()
app_test = pd.read_csv("application_test.csv")
print("Testing data shape: ", app_test.shape)
app_test.head()
app_train["TARGET"].value_counts()
app_train["TARGET"].astype(int).plot.hist()


def missing_values_table(df):
    # Total missing values
    mis_val = df.isnull().sum()
    mis_val_percent = 100 * df.isnull().sum() / len(df)
    mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
    mis_val_table_ren_columns = mis_val_table.rename(
        columns={0: "Missing Values", 1: "% of Total Values"}
    )
    mis_val_table_ren_columns = (
        mis_val_table_ren_columns[mis_val_table_ren_columns.iloc[:, 1] != 0]
        .sort_values("% of Total Values", ascending=False)
        .round(1)
    )
    print(
        "Your selected dataframe has " + str(df.shape[1]) + " columns.\n"
        "There are "
        + str(mis_val_table_ren_columns.shape[0])
        + " columns that have missing values."
    )
    return mis_val_table_ren_columns


app_train.dtypes.value_counts()
app_train.select_dtypes("object").apply(pd.Series.nunique, axis=0)
le = LabelEncoder()
le_count = 0
for col in app_train:
    if app_train[col].dtype == "object":
        # If 2 or fewer unique categories
        if len(list(app_train[col].unique())) <= 2:
            # Train on the training data
            le.fit(app_train[col])
            # Transform both training and testing data
            app_train[col] = le.transform(app_train[col])
            app_test[col] = le.transform(app_test[col])
            le_count += 1
print("%d columns were label encoded." % le_count)
app_train = pd.get_dummies(app_train)
app_test = pd.get_dummies(app_test)
print("Training Features shape: ", app_train.shape)
print("Testing Features shape: ", app_test.shape)
train_labels = app_train["TARGET"]
app_train, app_test = app_train.align(app_test, join="inner", axis=1)
app_train["TARGET"] = train_labels
print("Training Features shape: ", app_train.shape)
print("Testing Features shape: ", app_test.shape)
(app_train["DAYS_BIRTH"] / -365).describe()
app_train["DAYS_EMPLOYED"].describe()
app_train["DAYS_EMPLOYED"].plot.hist(title="Days Employment Histogram")
plt.xlabel("Days Employment")
anom = app_train[app_train["DAYS_EMPLOYED"] == 3]
non_anom = app_train[app_train["DAYS_EMPLOYED"] != 3]
print(
    "The non-anomalies default on %0.2f%% of loans" % (100 * non_anom["TARGET"].mean())
)
print("The anomalies default on %0.2f%% of loans" % (100 * anom["TARGET"].mean()))
print("There are %d anomalous days of employment" % len(anom))
app_train["DAYS_EMPLOYED_ANOM"] = app_train["DAYS_EMPLOYED"] == 3
app_train["DAYS_EMPLOYED"].replace({3: np.nan}, inplace=True)
app_train["DAYS_EMPLOYED"].plot.hist(title="Days Employment Histogram")
plt.xlabel("Days Employment")
app_test["DAYS_EMPLOYED_ANOM"] = app_test["DAYS_EMPLOYED"] == 3
app_test["DAYS_EMPLOYED"].replace({3: np.nan}, inplace=True)
print(
    "There are %d anomalies in the test data out of %d entries"
    % (app_test["DAYS_EMPLOYED_ANOM"].sum(), len(app_test))
)
correlations = app_train.corr()["TARGET"].sort_values()
print("Most Positive Correlations:\n", correlations.tail(15))
print("\nMost Negative Correlations:\n", correlations.head(15))
app_train["DAYS_BIRTH"] = abs(app_train["DAYS_BIRTH"])
app_train["DAYS_BIRTH"].corr(app_train["TARGET"])
plt.style.use("fivethirtyeight")
plt.hist(app_train["DAYS_BIRTH"] / 365, edgecolor="k", bins=25)
plt.title("Age of Client")
plt.xlabel("Age (years)")
plt.ylabel("Count")
plt.figure(figsize=(10, 8))
#
plt.xlabel("Age (years)")
plt.ylabel("Density")
plt.title("Distribution of Ages")
age_data = app_train[["TARGET", "DAYS_BIRTH"]]
age_data["YEARS_BIRTH"] = age_data["DAYS_BIRTH"] / 365
age_data["YEARS_BINNED"] = pd.cut(
    age_data["YEARS_BIRTH"], bins=np.linspace(20, 70, num=11)
)
age_data.head(10)
age_groups = age_data.groupby("YEARS_BINNED").mean()
age_groups
ext_data = app_train[
    ["TARGET", "EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]
]
ext_data_corrs = ext_data.corr()
ext_data_corrs
plt.figure(figsize=(8, 6))
sns.heatmap(ext_data_corrs, cmap=plt.cm.RdYlBu_r, vmin=-0.25, annot=True, vmax=0.6)
plt.title("Correlation Heatmap")
plot_data = ext_data.drop(columns=["DAYS_BIRTH"]).copy()
plot_data["YEARS_BIRTH"] = age_data["YEARS_BIRTH"]
plot_data = plot_data.dropna().loc[:100000, :]


def corr_func(x, y, **kwargs):
    r = np.corrcoef(x, y)[0][1]
    ax = plt.gca()
    ax.annotate("r = {:.2f}".format(r), xy=(0.2, 0.8), xycoords=ax.transAxes, size=20)


poly_features = app_train[
    ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH", "TARGET"]
]
poly_features_test = app_test[
    ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]
]
from sklearn.preprocessing import Imputer

imputer = Imputer(strategy="median")
poly_target = poly_features["TARGET"]
poly_features = poly_features.drop(columns=["TARGET"])
poly_features = imputer.fit_transform(poly_features)
poly_features_test = imputer.transform(poly_features_test)
from sklearn.preprocessing import PolynomialFeatures

poly_transformer = PolynomialFeatures(degree=3)
poly_transformer.fit(poly_features)
poly_features = poly_transformer.transform(poly_features)
poly_features_test = poly_transformer.transform(poly_features_test)
print("Polynomial Features shape: ", poly_features.shape)
poly_transformer.get_feature_names(
    input_features=["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]
)[:15]
poly_features = pd.DataFrame(
    poly_features,
    columns=poly_transformer.get_feature_names(
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]
    ),
)
poly_features["TARGET"] = poly_target
poly_corrs = poly_features.corr()["TARGET"].sort_values()
print(poly_corrs.head(10))
print(poly_corrs.tail(5))
poly_features_test = pd.DataFrame(
    poly_features_test,
    columns=poly_transformer.get_feature_names(
        ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3", "DAYS_BIRTH"]
    ),
)
poly_features["SK_ID_CURR"] = app_train["SK_ID_CURR"]
app_train_poly = app_train.merge(poly_features, on="SK_ID_CURR", how="left")
poly_features_test["SK_ID_CURR"] = app_test["SK_ID_CURR"]
app_test_poly = app_test.merge(poly_features_test, on="SK_ID_CURR", how="left")
app_train_poly, app_test_poly = app_train_poly.align(
    app_test_poly, join="inner", axis=1
)
print("Training data with polynomial features shape: ", app_train_poly.shape)
print("Testing data with polynomial features shape:  ", app_test_poly.shape)
app_train_domain = app_train.copy()
app_test_domain = app_test.copy()
app_train_domain["CREDIT_INCOME_PERCENT"] = (
    app_train_domain["AMT_CREDIT"] / app_train_domain["AMT_INCOME_TOTAL"]
)
app_train_domain["ANNUITY_INCOME_PERCENT"] = (
    app_train_domain["AMT_ANNUITY"] / app_train_domain["AMT_INCOME_TOTAL"]
)
app_train_domain["CREDIT_TERM"] = (
    app_train_domain["AMT_ANNUITY"] / app_train_domain["AMT_CREDIT"]
)
app_train_domain["DAYS_EMPLOYED_PERCENT"] = (
    app_train_domain["DAYS_EMPLOYED"] / app_train_domain["DAYS_BIRTH"]
)
app_test_domain["CREDIT_INCOME_PERCENT"] = (
    app_test_domain["AMT_CREDIT"] / app_test_domain["AMT_INCOME_TOTAL"]
)
app_test_domain["ANNUITY_INCOME_PERCENT"] = (
    app_test_domain["AMT_ANNUITY"] / app_test_domain["AMT_INCOME_TOTAL"]
)
app_test_domain["CREDIT_TERM"] = (
    app_test_domain["AMT_ANNUITY"] / app_test_domain["AMT_CREDIT"]
)
app_test_domain["DAYS_EMPLOYED_PERCENT"] = (
    app_test_domain["DAYS_EMPLOYED"] / app_test_domain["DAYS_BIRTH"]
)
from sklearn.preprocessing import Imputer, MinMaxScaler

if "TARGET" in app_train.columns:
    train = app_train.drop(columns=["TARGET"])
    # TODO (williamma12): Not sure why this line is necessary but it is
    app_test = app_test.drop(columns=["TARGET"])
else:
    train = app_train.copy()
features = list(train.columns)
test = app_test.copy()
imputer = Imputer(strategy="median")
scaler = MinMaxScaler(feature_range=(0, 1))
imputer.fit(train)
train = imputer.transform(train)
test = imputer.transform(app_test)
scaler.fit(train)
train = scaler.transform(train)
test = scaler.transform(test)
print("Training data shape: ", train.shape)
print("Testing data shape: ", test.shape)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(C=0.0001)
log_reg.fit(train, train_labels)
log_reg_pred = log_reg.predict_proba(test)[:, 1]
submit = app_test[["SK_ID_CURR"]]
submit["TARGET"] = log_reg_pred
submit.head()
submit.to_csv("log_reg_baseline.csv", index=False)
from sklearn.ensemble import RandomForestClassifier

random_forest = RandomForestClassifier(
    n_estimators=100, random_state=50, verbose=1, n_jobs=-1
)
random_forest.fit(train, train_labels)
feature_importance_values = random_forest.feature_importances_
feature_importances = pd.DataFrame(
    {"feature": features, "importance": feature_importance_values}
)
predictions = random_forest.predict_proba(test)[:, 1]
submit = app_test[["SK_ID_CURR"]]
submit["TARGET"] = predictions
submit.to_csv("random_forest_baseline.csv", index=False)
poly_features_names = list(app_train_poly.columns)
imputer = Imputer(strategy="median")
poly_features = imputer.fit_transform(app_train_poly)
poly_features_test = imputer.transform(app_test_poly)
scaler = MinMaxScaler(feature_range=(0, 1))
poly_features = scaler.fit_transform(poly_features)
poly_features_test = scaler.transform(poly_features_test)
random_forest_poly = RandomForestClassifier(
    n_estimators=100, random_state=50, verbose=1, n_jobs=-1
)
random_forest_poly.fit(poly_features, train_labels)
predictions = random_forest_poly.predict_proba(poly_features_test)[:, 1]
submit = app_test[["SK_ID_CURR"]]
submit["TARGET"] = predictions
submit.to_csv("random_forest_baseline_engineered.csv", index=False)
app_train_domain = app_train_domain.drop(columns="TARGET")
app_test_domain = app_test_domain.drop(columns="TARGET")
domain_features_names = list(app_train_domain.columns)
imputer = Imputer(strategy="median")
domain_features = imputer.fit_transform(app_train_domain)
domain_features_test = imputer.transform(app_test_domain)
scaler = MinMaxScaler(feature_range=(0, 1))
domain_features = scaler.fit_transform(domain_features)
domain_features_test = scaler.transform(domain_features_test)
random_forest_domain = RandomForestClassifier(
    n_estimators=100, random_state=50, verbose=1, n_jobs=-1
)
random_forest_domain.fit(domain_features, train_labels)
feature_importance_values_domain = random_forest_domain.feature_importances_
feature_importances_domain = pd.DataFrame(
    {"feature": domain_features_names, "importance": feature_importance_values_domain}
)
predictions = random_forest_domain.predict_proba(domain_features_test)[:, 1]
submit = app_test[["SK_ID_CURR"]]
submit["TARGET"] = predictions
submit.to_csv("random_forest_baseline_domain.csv", index=False)


def plot_feature_importances(df):
    df = df.sort_values("importance", ascending=False).reset_index()
    df["importance_normalized"] = df["importance"] / df["importance"].sum()
    plt.figure(figsize=(10, 6))
    ax = plt.subplot()
    ax.barh(
        list(reversed(list(df.index[:15]))),
        df["importance_normalized"].head(15),
        align="center",
        edgecolor="k",
    )
    ax.set_yticks(list(reversed(list(df.index[:15]))))
    ax.set_yticklabels(df["feature"].head(15))
    plt.xlabel("Normalized Importance")
    plt.title("Feature Importances")
    return df


feature_importances_sorted = plot_feature_importances(feature_importances)
feature_importances_domain_sorted = plot_feature_importances(feature_importances_domain)
import gc

import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold


def model(features, test_features, encoding="ohe", n_folds=5):
    test_ids = test_features["SK_ID_CURR"]
    labels = features["TARGET"]
    features = features.drop(columns=["SK_ID_CURR", "TARGET"])
    test_features = test_features.drop(columns=["SK_ID_CURR"])
    if encoding == "ohe":
        features = pd.get_dummies(features)
        test_features = pd.get_dummies(test_features)
        features, test_features = features.align(test_features, join="inner", axis=1)
        cat_indices = "auto"
    elif encoding == "le":
        label_encoder = LabelEncoder()
        cat_indices = []
        for i, col in enumerate(features):
            if features[col].dtype == "object":
                features[col] = label_encoder.fit_transform(
                    np.array(features[col].astype(str)).reshape((-1,))
                )
                test_features[col] = label_encoder.transform(
                    np.array(test_features[col].astype(str)).reshape((-1,))
                )
                cat_indices.append(i)
    else:
        raise ValueError("Encoding must be either 'ohe' or 'le'")
    print("Training Data Shape: ", features.shape)
    print("Testing Data Shape: ", test_features.shape)
    feature_names = list(features.columns)
    features = np.array(features)
    test_features = np.array(test_features)
    k_fold = KFold(n_splits=n_folds, shuffle=True, random_state=50)
    feature_importance_values = np.zeros(len(feature_names))
    test_predictions = np.zeros(test_features.shape[0])
    out_of_fold = np.zeros(features.shape[0])
    valid_scores = []
    train_scores = []
    for train_indices, valid_indices in k_fold.split(features):
        train_features, train_labels = features[train_indices], labels[train_indices]
        valid_features, valid_labels = features[valid_indices], labels[valid_indices]
        model = lgb.LGBMClassifier(
            n_estimators=10000,
            objective="binary",
            class_weight="balanced",
            learning_rate=0.05,
            reg_alpha=0.1,
            reg_lambda=0.1,
            subsample=0.8,
            n_jobs=-1,
            random_state=50,
        )
        model.fit(
            train_features,
            train_labels,
            eval_metric="auc",
            eval_set=[(valid_features, valid_labels), (train_features, train_labels)],
            eval_names=["valid", "train"],
            categorical_feature=cat_indices,
            early_stopping_rounds=100,
            verbose=200,
        )
        best_iteration = model.best_iteration_
        feature_importance_values += model.feature_importances_ / k_fold.n_splits
        test_predictions += (
            model.predict_proba(test_features, num_iteration=best_iteration)[:, 1]
            / k_fold.n_splits
        )
        out_of_fold[valid_indices] = model.predict_proba(
            valid_features, num_iteration=best_iteration
        )[:, 1]
        valid_score = model.best_score_["valid"]["auc"]
        train_score = model.best_score_["train"]["auc"]
        valid_scores.append(valid_score)
        train_scores.append(train_score)
        gc.enable()
        del model, train_features, valid_features
        gc.collect()
    submission = pd.DataFrame({"SK_ID_CURR": test_ids, "TARGET": test_predictions})
    feature_importances = pd.DataFrame(
        {"feature": feature_names, "importance": feature_importance_values}
    )
    valid_auc = roc_auc_score(labels, out_of_fold)
    valid_scores.append(valid_auc)
    train_scores.append(np.mean(train_scores))
    fold_names = list(range(n_folds))
    fold_names.append("overall")
    metrics = pd.DataFrame(
        {"fold": fold_names, "train": train_scores, "valid": valid_scores}
    )
    return submission, feature_importances, metrics


submission, fi, metrics = model(app_train, app_test)
print("Baseline metrics")
print(metrics)
fi_sorted = plot_feature_importances(fi)
submission.to_csv("baseline_lgb.csv", index=False)
app_train_domain["TARGET"] = train_labels
submission_domain, fi_domain, metrics_domain = model(app_train_domain, app_test_domain)
print("Baseline with domain knowledge features metrics")
print(metrics_domain)
fi_sorted = plot_feature_importances(fi_domain)
submission_domain.to_csv("baseline_lgb_domain_features.csv", index=False)
