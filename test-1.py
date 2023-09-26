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
)
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
# import shap

base = "result1"

s = pd.read_csv("./data/d-20230922-1.csv")

y = s["label"]
X = s.drop(columns=["label"])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

ids_train = X_train["id"]
ids_test = X_test["id"]
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])

sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)
X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)

oversample = RandomOverSampler()

X_train_std_s, y_train_s = oversample.fit_resample(X_train_std, y_train)

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
y_proba = classifier.predict_proba(X_test_std)

ids_test_0_0 = ids_test[(y_test == "ntb") & (y_pred == "ntb")]
ids_test_0_1 = ids_test[(y_test == "ntb") & (y_pred == "ntm")]
ids_test_0_2 = ids_test[(y_test == "ntb") & (y_pred == "tb")]
ids_test_1_0 = ids_test[(y_test == "ntm") & (y_pred == "ntb")]
ids_test_1_1 = ids_test[(y_test == "ntm") & (y_pred == "ntm")]
ids_test_1_2 = ids_test[(y_test == "ntm") & (y_pred == "tb")]
ids_test_2_0 = ids_test[(y_test == "tb") & (y_pred == "ntb")]
ids_test_2_1 = ids_test[(y_test == "tb") & (y_pred == "ntm")]
ids_test_2_2 = ids_test[(y_test == "tb") & (y_pred == "tb")]
ids_train_0 = ids_train[y_train == "ntb"]
ids_train_1 = ids_train[y_train == "ntm"]
ids_train_2 = ids_train[y_train == "tb"]

print("Accuracy: %.2f" % accuracy_score(y_test, y_pred))
print("Precision: %.2f" % precision_score(y_test, y_pred, average="macro"))
print("Recall: %.2f" % recall_score(y_test, y_pred, average="macro"))
print("F1: %.2f" % f1_score(y_test, y_pred, average="macro"))

confusion = confusion_matrix(y_test, y_pred)

label_names = ["ntb", "ntm", "tb"]
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"data/{base}/cm", dpi=300)
plt.close()

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, average="macro"),
    "recall": recall_score(y_test, y_pred, average="macro"),
    "f1": f1_score(y_test, y_pred, average="macro"),
}

metrics = pd.DataFrame(metrics, index=[0])
metrics.to_csv(f"data/{base}/metrics.csv", index=False)


s_ntb = pd.read_csv('./data/结核数据/非-TB-Table 1.csv')
s_ntm = pd.read_csv('./data/结核数据/NTM-Table 1.csv')
s_tb = pd.read_csv('./data/结核数据/TB-Table 1.csv')
s_ntb['label'] = 'ntb'
s_ntm['label'] = 'ntm'
s_tb['label'] = 'tb'
ori = pd.concat([s_ntb, s_ntm, s_tb], axis=0)
ori.dropna()
ori.reset_index(drop=True, inplace=True)

def result_row(row):
    if row.name in ids_train_0.values:
        return "Train NTB"
    if row.name in ids_train_1.values:
        return "Train NTM"
    if row.name in ids_train_2.values:
        return "Train TB"
    if row.name in ids_test_0_0.values:
        return "Test NTB -> NTB"
    if row.name in ids_test_0_1.values:
        return "Test NTB -> NTM"
    if row.name in ids_test_0_2.values:
        return "Test NTB -> TB"
    if row.name in ids_test_1_0.values:
        return "Test NTM -> NTB"
    if row.name in ids_test_1_1.values:
        return "Test NTM -> NTM"
    if row.name in ids_test_1_2.values:
        return "Test NTM -> TB"
    if row.name in ids_test_2_0.values:
        return "Test TB -> NTB"
    if row.name in ids_test_2_1.values:
        return "Test TB -> NTM"
    if row.name in ids_test_2_2.values:
        return "Test TB -> TB"

ori['result'] = ori.apply(lambda row: result_row(row), axis=1)

ori["proba_ntb"] = '-'
ori["proba_ntm"] = '-'
ori["proba_tb"] = '-'

for idx, test_id in enumerate(ids_test):
    ori.loc[test_id, 'proba_ntb'] = y_proba[idx][0]
    ori.loc[test_id, 'proba_ntm'] = y_proba[idx][1]
    ori.loc[test_id, 'proba_tb'] = y_proba[idx][2]

ori.to_csv(f"data/{base}/result.csv", index=False)

# explainer = shap.KernelExplainer(classifier.predict_proba, X_train_std_s)
# shap_values = explainer.shap_values(X_test_std, nsamples=100)
# shap.initjs()
# shap.summary_plot(shap_values[1], X_test_std, show=False)
# plt.savefig(f"data/{base}/shap", dpi=300)
# plt.close()
