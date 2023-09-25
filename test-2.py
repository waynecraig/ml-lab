import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
)
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.utils import resample
import shap

base = "result2"

s = pd.read_csv("./data/d-20230922-2.csv")

y = s["label"]
X = s.drop(columns=["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

ids_train = X_train["id"]
ids_test = X_test["id"]
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])
X = X.drop(columns=["id"])

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_train_std = pd.DataFrame(X_train_std, columns=X.columns)
X_test_std = pd.DataFrame(X_test_std, columns=X.columns)

oversample = RandomOverSampler()

X_train_std_s, y_train_s = oversample.fit_resample(X_train_std, y_train)
X_train_std_s = pd.DataFrame(X_train_std_s, columns=X.columns)

lr = LogisticRegression(C=100.0, random_state=71, max_iter=500)
rf = RandomForestClassifier(
    n_estimators=100, max_depth=2, random_state=71
)
svm = SVC(kernel="linear", C=1.0, random_state=71, probability=True)

classifier = VotingClassifier(
    estimators=[("lr", lr), ("rf", rf), ("svm", svm)],
    voting="soft",
)

classifier.fit(X_train_std_s, y_train_s)

y_pred = classifier.predict(X_test_std)

ids_test_0_0 = ids_test[(y_test == 0) & (y_pred == 0)]
ids_test_0_1 = ids_test[(y_test == 0) & (y_pred == 1)]
ids_test_1_0 = ids_test[(y_test == 1) & (y_pred == 0)]
ids_test_1_1 = ids_test[(y_test == 1) & (y_pred == 1)]
ids_train_0 = ids_train[y_train == 0]
ids_train_1 = ids_train[y_train == 1]

print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision: %.2f" % precision_score(y_test, y_pred))
print("Recall: %.2f" % recall_score(y_test, y_pred))
print("F1: %.2f" % f1_score(y_test, y_pred))

confusion = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = confusion.ravel()
specificity = tn / (tn + fp)
sensitivity = tp / (tp + fn)

print("Specificity: %.2f" % specificity)
print("Sensitivity: %.2f" % sensitivity)

# draw ROC curve
fpr, tpr, _ = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

tprs = []
aucs = []
n_bootstraps = 1000
rng = np.random.RandomState(0)
for i in range(n_bootstraps):
    y_test_bootstrap, y_pred_bootstrap = resample(y_test, y_pred, random_state=i)
    fpr_bootstrap, tpr_bootstrap, _ = roc_curve(y_test_bootstrap, y_pred_bootstrap)
    tprs.append(np.interp(fpr, fpr_bootstrap, tpr_bootstrap))
    tprs[-1][0] = 0.0
    roc_auc_bootstrap = auc(fpr_bootstrap, tpr_bootstrap)
    aucs.append(roc_auc_bootstrap)

tprs = np.array(tprs)
mean_tprs = tprs.mean(axis=0)
std = tprs.std(axis=0)
lower_bound = np.percentile(aucs, 2.5)
upper_bound = np.percentile(aucs, 97.5)

plt.figure()
plt.plot(fpr, tpr, color="b", label=f"ROC curve (AUC = {roc_auc:.2f})")
plt.fill_between(
    fpr,
    mean_tprs - 1.96 * std,
    mean_tprs + 1.96 * std,
    color="grey",
    alpha=0.3,
    label=f"95% CI ({lower_bound:.2f}-{upper_bound:.2f})",
)
plt.plot([0, 1], [0, 1], color="r", linestyle="--")
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.savefig(f"data/{base}/roc_with_ci", dpi=300)
plt.close()

label_names = ["other", "tb"]
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"data/{base}/cm", dpi=300)
plt.close()

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred),
    "recall": recall_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "specificity": specificity,
    "sensitivity": sensitivity,
    "roc_auc": roc_auc,
    "auc_ci_lower": lower_bound,
    "auc_ci_upper": upper_bound,
}

metrics = pd.DataFrame(metrics, index=[0])
metrics.to_csv(f"data/{base}/metrics.csv", index=False)


s_ntb = pd.read_csv('./data/结核数据/非-TB-Table 1.csv')
s_ntm = pd.read_csv('./data/结核数据/NTM-Table 1.csv')
s_tb = pd.read_csv('./data/结核数据/TB-Table 1.csv')
s_ntb['label'] = 0
s_ntm['label'] = 0
s_tb['label'] = 1
ori = pd.concat([s_ntb, s_ntm, s_tb], axis=0)
ori.dropna()
ori.reset_index(drop=True, inplace=True)

def result_row(row):
    if row.name in ids_train_0.values:
        return "Train OTHER"
    if row.name in ids_train_1.values:
        return "Train TB"
    if row.name in ids_test_0_0.values:
        return "Test OTHER -> OTHER"
    if row.name in ids_test_0_1.values:
        return "Test OTHER -> TB"
    if row.name in ids_test_1_0.values:
        return "Test TB -> OTHER"
    if row.name in ids_test_1_1.values:
        return "Test TB -> TB"

ori['result'] = ori.apply(lambda row: result_row(row), axis=1)

ori.to_csv(f"data/{base}/result.csv", index=False)

explainer = shap.KernelExplainer(classifier.predict_proba, X_train_std_s)
shap_values = explainer.shap_values(X_test_std, nsamples=100)

shap.initjs()
shap.summary_plot(shap_values[1], X_test_std, show=False)
plt.savefig(f"data/{base}/shap", dpi=300)
plt.close()