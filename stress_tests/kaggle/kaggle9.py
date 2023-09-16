import matplotlib

matplotlib.use("PS")
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import skew

import modin.pandas as pd

train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")
train.head()
all_data = pd.concat(
    (
        train.loc[:, "MSSubClass":"SaleCondition"],
        test.loc[:, "MSSubClass":"SaleCondition"],
    )
)
matplotlib.rcParams["figure.figsize"] = (12.0, 6.0)
prices = pd.DataFrame(
    {"price": train["SalePrice"], "log(price + 1)": np.log1p(train["SalePrice"])}
)
prices.hist()
train["SalePrice"] = np.log1p(train["SalePrice"])
numeric_feats = all_data.dtypes[all_data.dtypes != "object"].index
skewed_feats = train[numeric_feats].apply(
    lambda x: skew(x.dropna())
)  # compute skewness
skewed_feats = skewed_feats[skewed_feats > 0.75]
skewed_feats = skewed_feats.index
all_data[skewed_feats] = np.log1p(all_data[skewed_feats])
all_data = pd.get_dummies(all_data)
all_data = all_data.fillna(all_data.mean())
X_train = all_data[: train.shape[0]]
X_test = all_data[train.shape[0] :]
y = train.SalePrice
from sklearn.linear_model import LassoCV  # RidgeCV, ElasticNet, LassoLarsCV
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_score


def rmse_cv(model):
    rmse = np.sqrt(
        -cross_val_score(model, X_train, y, scoring="neg_mean_squared_error", cv=5)
    )
    return rmse


model_ridge = Ridge()
alphas = [0.05, 0.1, 0.3, 1, 3, 5, 10, 15, 30, 50, 75]
cv_ridge = [rmse_cv(Ridge(alpha=alpha)).mean() for alpha in alphas]
cv_ridge = pd.Series(cv_ridge, index=alphas)
cv_ridge.plot(title="Validation - Just Do It")
plt.xlabel("alpha")
plt.ylabel("rmse")
cv_ridge.min()
model_lasso = LassoCV(alphas=[1, 0.1, 0.001, 0.0005]).fit(X_train, y)
rmse_cv(model_lasso).mean()
coef = pd.Series(model_lasso.coef_, index=X_train.columns)
print(
    "Lasso picked "
    + str(sum(coef != 0))
    + " variables and eliminated the other "
    + str(sum(coef == 0))
    + " variables"
)
imp_coef = pd.concat([coef.sort_values().head(10), coef.sort_values().tail(10)])
matplotlib.rcParams["figure.figsize"] = (8.0, 10.0)
imp_coef.plot(kind="barh")
plt.title("Coefficients in the Lasso Model")
matplotlib.rcParams["figure.figsize"] = (6.0, 6.0)
preds = pd.DataFrame({"preds": model_lasso.predict(X_train), "true": y})
preds["residuals"] = preds["true"] - preds["preds"]
preds.plot(x="preds", y="residuals", kind="scatter")
import xgboost as xgb

dtrain = xgb.DMatrix(X_train, label=y)
dtest = xgb.DMatrix(X_test)
params = {"max_depth": 2, "eta": 0.1}
model = xgb.cv(params, dtrain, num_boost_round=500, early_stopping_rounds=100)
model.loc[30:, ["test-rmse-mean", "train-rmse-mean"]].plot()
model_xgb = xgb.XGBRegressor(
    n_estimators=360, max_depth=2, learning_rate=0.1
)  # the params were tuned using xgb.cv
model_xgb.fit(X_train, y)
xgb_preds = np.expm1(model_xgb.predict(X_test))
lasso_preds = np.expm1(model_lasso.predict(X_test))
predictions = pd.DataFrame({"xgb": xgb_preds, "lasso": lasso_preds})
predictions.plot(x="xgb", y="lasso", kind="scatter")
preds = 0.7 * lasso_preds + 0.3 * xgb_preds
solution = pd.DataFrame({"id": test.Id, "SalePrice": preds})
solution.to_csv("ridge_sol.csv", index=False)
from keras.layers import Dense
from keras.models import Sequential
from keras.regularizers import l1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

X_train = StandardScaler().fit_transform(X_train)
X_tr, X_val, y_tr, y_val = train_test_split(X_train, y, random_state=3)
X_tr.shape
X_tr
model = Sequential()
model.add(Dense(1, input_dim=X_train.shape[1], W_regularizer=l1(0.001)))
model.compile(loss="mse", optimizer="adam")
model.summary()
hist = model.fit(X_tr, y_tr, validation_data=(X_val, y_val))
pd.Series(model.predict(X_val)[:, 0]).hist()
