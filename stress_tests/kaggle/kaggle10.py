import matplotlib

matplotlib.use("PS")
import warnings

import matplotlib.pyplot as plt
import numpy as np  # linear algebra
import seaborn as sns

import modin.pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

warnings.filterwarnings("ignore")
data = pd.read_csv("column_2C_weka.csv")
print(plt.style.available)  # look at available plot styles
plt.style.use("ggplot")
data.head()
data.info()
data.describe()
color_list = ["red" if i == "Abnormal" else "green" for i in data.loc[:, "class"]]
pd.plotting.scatter_matrix(
    data.loc[:, data.columns != "class"],
    c=color_list,
    figsize=[15, 15],
    diagonal="hist",
    alpha=0.5,
    s=200,
    marker="*",
    edgecolor="black",
)
plt.show()
sns.countplot(x="class", data=data)
data.loc[:, "class"].value_counts()
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=3)
x, y = data.loc[:, data.columns != "class"], data.loc[:, "class"]
knn.fit(x, y)
prediction = knn.predict(x)
print("Prediction: {}".format(prediction))
from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
knn = KNeighborsClassifier(n_neighbors=3)
x, y = data.loc[:, data.columns != "class"], data.loc[:, "class"]
knn.fit(x_train, y_train)
prediction = knn.predict(x_test)
print("With KNN (K=3) accuracy is: ", knn.score(x_test, y_test))  # accuracy
neig = np.arange(1, 25)
train_accuracy = []
test_accuracy = []
for i, k in enumerate(neig):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x_train, y_train)
    train_accuracy.append(knn.score(x_train, y_train))
    test_accuracy.append(knn.score(x_test, y_test))
plt.figure(figsize=[13, 8])
plt.plot(neig, test_accuracy, label="Testing Accuracy")
plt.plot(neig, train_accuracy, label="Training Accuracy")
plt.legend()
plt.title("-value VS Accuracy")
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")
plt.xticks(neig)
plt.savefig("graph.png")
plt.show()
print(
    "Best accuracy is {} with K = {}".format(
        np.max(test_accuracy), 1 + test_accuracy.index(np.max(test_accuracy))
    )
)
data1 = data[data["class"] == "A"]
x = np.array(data1.loc[:, "pelvic_incidence"]).reshape(-1, 1)
y = np.array(data1.loc[:, "sacral_slope"]).reshape(-1, 1)
plt.figure(figsize=[10, 10])
plt.scatter(x=x, y=y)
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.show()
from sklearn.linear_model import LinearRegression

reg = LinearRegression()
predict_space = np.linspace(min(x), max(x)).reshape(-1, 1)
reg.fit(x, y)
predicted = reg.predict(predict_space)
print("R^2 score: ", reg.score(x, y))
plt.plot(predict_space, predicted, color="black", linewidth=3)
plt.scatter(x=x, y=y)
plt.xlabel("pelvic_incidence")
plt.ylabel("sacral_slope")
plt.show()
from sklearn.model_selection import cross_val_score

reg = LinearRegression()
k = 5
cv_result = cross_val_score(reg, x, y, cv=k)  # uses R^2 as score
print("CV Scores: ", cv_result)
print("CV scores average: ", np.sum(cv_result) / k)
from sklearn.linear_model import Ridge

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=2, test_size=0.3)
ridge = Ridge(alpha=0.1, normalize=True)
ridge.fit(x_train, y_train)
ridge_predict = ridge.predict(x_test)
print("Ridge score: ", ridge.score(x_test, y_test))
from sklearn.linear_model import Lasso

x = np.array(
    data1.loc[
        :,
        [
            "pelvic_incidence",
            "pelvic_tilt numeric",
            "lumbar_lordosis_angle",
            "pelvic_radius",
        ],
    ]
)
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=3, test_size=0.3)
lasso = Lasso(alpha=0.1, normalize=True)
lasso.fit(x_train, y_train)
ridge_predict = lasso.predict(x_test)
print("Lasso score: ", lasso.score(x_test, y_test))
print("Lasso coefficients: ", lasso.coef_)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

x, y = data.loc[:, data.columns != "class"], data.loc[:, "class"]
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=1)
rf = RandomForestClassifier(random_state=4)
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix: \n", cm)
print("Classification report: \n", classification_report(y_test, y_pred))
sns.heatmap(cm, annot=True, fmt="d")
plt.show()
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_curve

data["class_binary"] = [1 if i == "Abnormal" else 0 for i in data.loc[:, "class"]]
x, y = (
    data.loc[:, (data.columns != "class") & (data.columns != "class_binary")],
    data.loc[:, "class_binary"],
)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=42
)
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred_prob = logreg.predict_proba(x_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.plot([0, 1], [0, 1], "k--")
plt.plot(fpr, tpr)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC")
plt.show()
from sklearn.model_selection import GridSearchCV

grid = {"n_neighbors": np.arange(1, 50)}
knn = KNeighborsClassifier()
knn_cv = GridSearchCV(knn, grid, cv=3)  # GridSearchCV
knn_cv.fit(x, y)  # Fit
print("Tuned hyperparameter k: {}".format(knn_cv.best_params_))
print("Best score: {}".format(knn_cv.best_score_))
param_grid = {"C": np.logspace(-3, 3, 7), "penalty": ["l1", "l2"]}
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.3, random_state=12
)
logreg = LogisticRegression()
logreg_cv = GridSearchCV(logreg, param_grid, cv=3)
logreg_cv.fit(x_train, y_train)
print("Tuned hyperparameters : {}".format(logreg_cv.best_params_))
print("Best Accuracy: {}".format(logreg_cv.best_score_))
data = pd.read_csv("column_2C_weka.csv")
df = pd.get_dummies(data)
df.head(10)
df.drop("class_Normal", axis=1, inplace=True)
df.head(10)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

steps = [("scalar", StandardScaler()), ("SVM", SVC())]
pipeline = Pipeline(steps)
parameters = {"SVM__C": [1, 10, 100], "SVM__gamma": [0.1, 0.01]}
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)
cv = GridSearchCV(pipeline, param_grid=parameters, cv=3)
cv.fit(x_train, y_train)
y_pred = cv.predict(x_test)
print("Accuracy: {}".format(cv.score(x_test, y_test)))
print("Tuned Model Parameters: {}".format(cv.best_params_))
data = pd.read_csv("column_2C_weka.csv")
plt.scatter(data["pelvic_radius"], data["degree_spondylolisthesis"])
plt.xlabel("pelvic_radius")
plt.ylabel("degree_spondylolisthesis")
plt.show()
data2 = data.loc[:, ["degree_spondylolisthesis", "pelvic_radius"]]
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters=2)
kmeans.fit(data2)
labels = kmeans.predict(data2)
plt.scatter(data["pelvic_radius"], data["degree_spondylolisthesis"], c=labels)
plt.xlabel("pelvic_radius")
plt.xlabel("degree_spondylolisthesis")
plt.show()
df = pd.DataFrame({"labels": labels, "class": data["class"]})
ct = pd.crosstab(df["labels"], df["class"])
print(ct)
inertia_list = np.empty(8)
for i in range(1, 8):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(data2)
    inertia_list[i] = kmeans.inertia_
plt.plot(range(0, 8), inertia_list, "-o")
plt.xlabel("Number of cluster")
plt.ylabel("Inertia")
plt.show()
data = pd.read_csv("column_2C_weka.csv")
data3 = data.drop("class", axis=1)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

scalar = StandardScaler()
kmeans = KMeans(n_clusters=2)
pipe = make_pipeline(scalar, kmeans)
pipe.fit(data3)
labels = pipe.predict(data3)
df = pd.DataFrame({"labels": labels, "class": data["class"]})
ct = pd.crosstab(df["labels"], df["class"])
print(ct)
from scipy.cluster.hierarchy import dendrogram, linkage

merg = linkage(data3.iloc[200:220, :], method="single")
dendrogram(merg, leaf_rotation=90, leaf_font_size=6)
plt.show()
from sklearn.manifold import TSNE

model = TSNE(learning_rate=100)
transformed = model.fit_transform(data2)
x = transformed[:, 0]
y = transformed[:, 1]
plt.scatter(x, y, c=color_list)
plt.xlabel("pelvic_radius")
plt.xlabel("degree_spondylolisthesis")
plt.show()
from sklearn.decomposition import PCA

model = PCA()
model.fit(data3)
transformed = model.transform(data3)
print("Principle components: ", model.components_)
scaler = StandardScaler()
pca = PCA()
pipeline = make_pipeline(scaler, pca)
pipeline.fit(data3)
plt.bar(range(pca.n_components_), pca.explained_variance_)
plt.xlabel("PCA feature")
plt.ylabel("variance")
plt.show()
pca = PCA(n_components=2)
pca.fit(data3)
transformed = pca.transform(data3)
x = transformed[:, 0]
y = transformed[:, 1]
plt.scatter(x, y, c=color_list)
plt.show()
