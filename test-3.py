import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    auc,
    precision_score,
    recall_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
from imblearn.over_sampling import RandomOverSampler
import seaborn as sns
import matplotlib.pyplot as plt
import shap
import numpy as np

base = "result3"

s = pd.read_csv("./data/d-20230926.csv")

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

classifier = RandomForestClassifier(
    n_estimators=100, max_depth=2, random_state=71
)

classifier.fit(X_train_std_s, y_train_s)

y_pred = classifier.predict(X_test_std)
y_proba = classifier.predict_proba(X_test_std)

ids_test_0_0 = ids_test[(y_test == "atb") & (y_pred == "atb")]
ids_test_0_1 = ids_test[(y_test == "atb") & (y_pred == "cdtb")]
ids_test_0_2 = ids_test[(y_test == "atb") & (y_pred == "ntb")]
ids_test_0_3 = ids_test[(y_test == "atb") & (y_pred == "ntm")]
ids_test_1_0 = ids_test[(y_test == "cdtb") & (y_pred == "atb")]
ids_test_1_1 = ids_test[(y_test == "cdtb") & (y_pred == "cdtb")]
ids_test_1_2 = ids_test[(y_test == "cdtb") & (y_pred == "ntb")]
ids_test_1_3 = ids_test[(y_test == "cdtb") & (y_pred == "ntm")]
ids_test_2_0 = ids_test[(y_test == "ntb") & (y_pred == "atb")]
ids_test_2_1 = ids_test[(y_test == "ntb") & (y_pred == "cdtb")]
ids_test_2_2 = ids_test[(y_test == "ntb") & (y_pred == "ntb")]
ids_test_2_3 = ids_test[(y_test == "ntb") & (y_pred == "ntm")]
ids_test_3_0 = ids_test[(y_test == "ntm") & (y_pred == "atb")]
ids_test_3_1 = ids_test[(y_test == "ntm") & (y_pred == "cdtb")]
ids_test_3_2 = ids_test[(y_test == "ntm") & (y_pred == "ntb")]
ids_test_3_3 = ids_test[(y_test == "ntm") & (y_pred == "ntm")]
ids_train_0 = ids_train[y_train == "atb"]
ids_train_1 = ids_train[y_train == "cdtb"]
ids_train_2 = ids_train[y_train == "ntb"]
ids_train_3 = ids_train[y_train == "ntm"]


accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")

precisions = precision_score(y_test, y_pred, average=None)
recalls = recall_score(y_test, y_pred, average=None)
f1s = f1_score(y_test, y_pred, average=None)

precision_macro = precision_score(y_test, y_pred, average="macro")
precision_micro = precision_score(y_test, y_pred, average="micro")
precision_weighted = precision_score(y_test, y_pred, average="weighted")
recall_macro = recall_score(y_test, y_pred, average="macro")
recall_micro = recall_score(y_test, y_pred, average="micro")
recall_weighted = recall_score(y_test, y_pred, average="weighted")
f1_macro = f1_score(y_test, y_pred, average="macro")
f1_micro = f1_score(y_test, y_pred, average="micro")
f1_weighted = f1_score(y_test, y_pred, average="weighted")


metrics = pd.DataFrame(
    {
        "Metrics": ["atb", "cdtb", "ntb", "ntm", "Macro Average", "Micro Average", "Weighted Average"],
        "Precision": [*precisions, precision_macro, precision_micro, precision_weighted],
        "Recall": [*recalls, recall_macro, recall_micro, recall_weighted],
        "F1-Score": [*f1s, f1_macro, f1_micro, f1_weighted],
    }
)

metrics.to_csv(f"data/{base}/metrics.csv", index=False)


for idx, label in enumerate(classifier.classes_):
    y_score = y_proba[:, idx]
    precision, recall, _ = precision_recall_curve(
        y_test, y_score, pos_label=label
    )
    plt.plot(recall, precision, label=label)

plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.savefig(f"data/{base}/prc", dpi=300)
plt.close()


confusion = confusion_matrix(y_test, y_pred)

label_names = ["atb", "cdtb", "ntb", "ntm"]
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"data/{base}/cm", dpi=300)
plt.close()


s_atb = pd.read_csv('./data/结核数据20230926/ATB-Table 1.csv')
s_cdtb = pd.read_csv('./data/结核数据20230926/CDTB-Table 1.csv')
s_ntb = pd.read_csv('./data/结核数据20230926/NTB-Table 1.csv')
s_ntm = pd.read_csv('./data/结核数据20230926/NTM-Table 1.csv')

s_atb['label'] = 'atb'
s_cdtb['label'] = 'cdtb'
s_ntb['label'] = 'ntb'
s_ntm['label'] = 'ntm'

ori = pd.concat([s_atb, s_cdtb, s_ntb, s_ntm], axis=0)
ori.dropna()
ori.reset_index(drop=True, inplace=True)

def result_row(row):
    if row.name in ids_train_0.values:
        return "Train ATB"
    if row.name in ids_train_1.values:
        return "Train CDTB"
    if row.name in ids_train_2.values:
        return "Train NTB"
    if row.name in ids_train_3.values:
        return "Train NTM"
    if row.name in ids_test_0_0.values:
        return "Test ATB -> ATB"
    if row.name in ids_test_0_1.values:
        return "Test ATB -> CDTB"
    if row.name in ids_test_0_2.values:
        return "Test ATB -> NTB"
    if row.name in ids_test_0_3.values:
        return "Test ATB -> NTM"
    if row.name in ids_test_1_0.values:
        return "Test CDTB -> ATB"
    if row.name in ids_test_1_1.values:
        return "Test CDTB -> CDTB"
    if row.name in ids_test_1_2.values:
        return "Test CDTB -> NTB"
    if row.name in ids_test_1_3.values:
        return "Test CDTB -> NTM"
    if row.name in ids_test_2_0.values:
        return "Test NTB -> ATB"
    if row.name in ids_test_2_1.values:
        return "Test NTB -> CDTB"
    if row.name in ids_test_2_2.values:
        return "Test NTB -> NTB"
    if row.name in ids_test_2_3.values:
        return "Test NTB -> NTM"
    if row.name in ids_test_3_0.values:
        return "Test NTM -> ATB"
    if row.name in ids_test_3_1.values:
        return "Test NTM -> CDTB"
    if row.name in ids_test_3_2.values:
        return "Test NTM -> NTB"
    if row.name in ids_test_3_3.values:
        return "Test NTM -> NTM"

ori['result'] = ori.apply(lambda row: result_row(row), axis=1)

ori["proba_atb"] = '-'
ori["proba_cdtb"] = '-'
ori["proba_ntb"] = '-'
ori["proba_ntm"] = '-'

for idx, test_id in enumerate(ids_test):
    ori.loc[test_id, 'proba_atb'] = y_proba[idx][0]
    ori.loc[test_id, 'proba_cdtb'] = y_proba[idx][1]
    ori.loc[test_id, 'proba_ntb'] = y_proba[idx][2]
    ori.loc[test_id, 'proba_ntm'] = y_proba[idx][3]

ori.to_csv(f"data/{base}/result.csv", index=False)

# explainer = shap.TreeExplainer(classifier)
# shap_values = explainer.shap_values(X_test_std)
# shap.initjs()
# shap.summary_plot(shap_values[1], X_test_std, show=False)
# plt.savefig(f"data/{base}/shap", dpi=300)
# plt.close()

# for i in range(0, len(X_test_std)):
#     id = ids_test.iloc[i]
#     shap.force_plot(
#         explainer.expected_value[1],
#         shap_values[1][i],
#         feature_names=X_test.columns,
#         matplotlib=True,
#         show=False,
#     )
#     plt.savefig(f"data/{base}/shap/force-{id+2}", dpi=300)
#     plt.close()




# recall = sensitivity = TP / (TP + FN)
# specificity = TN / (TN + FP)
# precision = TP / (TP + FP)
# f1 = 2 * precision * recall / (precision + recall)

fpr = {}
tpr = {}
auc_score = {}
for i in range(classifier.n_classes_):
    y_true_i = (y_test == classifier.classes_[i])
    y_proba_i = y_proba[:, i]
    fpr[i], tpr[i], _ = roc_curve(y_true_i, y_proba_i, pos_label=i)
    auc_score[i] = auc(fpr[i], tpr[i])

    TP = np.sum(y_true_i & (y_proba_i >= 0.5))
    FP = np.sum(~y_true_i & (y_proba_i >= 0.5))
    TN = np.sum(~y_true_i & (y_proba_i < 0.5))
    FN = np.sum(y_true_i & (y_proba_i < 0.5))
    sensitivity = TP / (TP + FN)
    specificity = TN / (TN + FP)
    print(f"{classifier.classes_[i]}: sensitivity={sensitivity:.4f}, specificity={specificity:.4f}")


# def calculate_dca(curve, threshold):
#     EU = curve - threshold
    
#     net_benefit = EU / (1 - threshold)

#     return net_benefit


# def plot_dca_curve(thresholds, curve):
#     net_benefits = []

#     for threshold in thresholds:
#         net_benefit = calculate_dca(curve, threshold)
#         net_benefits.append(net_benefit)

#     plt.plot(thresholds, net_benefits)
#     plt.axhline(0, color='gray', linestyle='--')
#     plt.xlabel('Threshold')
#     plt.ylabel('Net Benefit')
#     plt.title('DCA Curve')
#     plt.grid(True)

# # 计算DCA曲线需要的结果
# n_classes = len(classifier.classes_)
# thresholds = np.linspace(0, 1, num=100)
# curve = np.zeros_like(thresholds)

# for k in range(n_classes):
#     y_true_k = (y_test == classifier.classes_[k])
#     y_pred_proba_k = y_proba[:, k]
    
#     # 计算True Positive Rate 和 False Positive Rate
#     TP = np.sum(y_true_k & (y_pred_proba_k >= thresholds))
#     FP = np.sum(~y_true_k & (y_pred_proba_k >= thresholds))
#     TN = np.sum(~y_true_k & (y_pred_proba_k < thresholds))
#     FN = np.sum(y_true_k & (y_pred_proba_k < thresholds))

#     # 计算敏感性和特异性
#     sensitivity = TP / (TP + FN)
#     specificity = TN / (TN + FP)

#     # 计算DCA曲线
#     curve += (sensitivity - specificity) / n_classes

# # 绘制DCA曲线
# plot_dca_curve(thresholds, curve)
# plt.savefig(f"data/{base}/dca", dpi=300)
# plt.close()