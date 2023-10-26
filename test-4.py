# Plan
# 1. Read data from csv file.
# 2. Get X and y. X is the features, y is the labels.
# 3. Split the data into training set and test set.
# 4. Standardize the features.
# 5. Oversample the training set, because the data is imbalanced.
# 6. Optimize the hyperparameters of the Random Forest Classifier.
# 7. Initialize the classifier and train it.
# 8. Evaluate the classifier.
# 9. Calculate the probabilities of the test set.
# 10. Save the result and probabilities to csv file.
# 11. Calculate the metrics and save them to csv file.
# 12. Plot the confusion matrix.
# 13. Plot the ROC curve.
# 14. Plot the PR curve.
# 15. Plot the feature importance.
# 16. Plot the decision tree.
# 17. Plot the SHAP values.
# 18. Plot the DCA curve (decision curve analysis).

import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    auc,
    precision_score,
    recall_score,
    accuracy_score,
    make_scorer,
    f1_score,
    confusion_matrix,
    precision_recall_curve,
    roc_curve,
)
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
import shap
import matplotlib.ticker as mtick


base = "result4"

# 1. Read data from csv file.
s = pd.read_csv("./data/d-20231016.csv")

# 2. Get X and y. X is the features, y is the labels.
y = s["label"]
X = s.drop(columns=["label"])

# 3. Split the data into training set and test set.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=37, stratify=y
)

ids_train = X_train["id"]
ids_test = X_test["id"]
X_train = X_train.drop(columns=["id"])
X_test = X_test.drop(columns=["id"])

# 4. Standardize the features.
sc = StandardScaler()
sc.fit(X_train)
X_train_std = sc.transform(X_train)
X_test_std = sc.transform(X_test)
X_train_std = pd.DataFrame(X_train_std, columns=X_train.columns)
X_test_std = pd.DataFrame(X_test_std, columns=X_test.columns)


# 5. Oversample the training set, because the data is imbalanced.
oversample = RandomOverSampler()
X_train_std_s, y_train_s = oversample.fit_resample(X_train_std, y_train)

# 6. Optimize the hyperparameters of the Random Forest Classifier.
# param_grid = {
#   'n_estimators': [100, 200, 300],
#   'max_depth': [None, 5, 10],
#   'min_samples_split': [2, 5, 10],
#   'min_samples_leaf': [1, 2, 4]
# }
# rf = RandomForestClassifier()
# gscv = GridSearchCV(rf, param_grid, cv=5, scoring="accuracy", n_jobs=-1)
# gscv.fit(X_train_std_s, y_train_s)

# best_params = gscv.best_params_
# print(best_params)
best_params = {
    "max_depth": None,
    "min_samples_leaf": 1,
    "min_samples_split": 5,
    "n_estimators": 300,
}

# 7. Initialize the classifier and train it.
best_rf = RandomForestClassifier(**best_params)
best_rf.fit(X_train_std_s, y_train_s)

# 8. Evaluate the classifier.
y_pred = best_rf.predict(X_test_std)

# 9. Calculate the probabilities of the test set.
y_proba = best_rf.predict_proba(X_test_std)

# 10. Save the result and probabilities to csv file.
cols = [
    "住院号",
    "性别",
    "年龄",
    "咳嗽",
    "发热",
    "咯血",
    "肺部CT",
    "IGRAs-CZ",
    "IGRAs",
    "TBC涂片",
    "TB-DNA",
]
s_aptb = pd.read_csv("./data/结核数据20231016/APTB-Table 1.csv", usecols=cols)
s_cdptb = pd.read_csv("./data/结核数据20231016/CDPTB-Table 1.csv", usecols=cols)
s_itb = pd.read_csv("./data/结核数据20231016/ITB-Table 1.csv", usecols=cols)
s_n_tpd = pd.read_csv("./data/结核数据20231016/N-TPD-Table 1.csv", usecols=cols)

s_aptb["label"] = "APTB"
s_cdptb["label"] = "CDPTB"
s_itb["label"] = "ITB"
s_n_tpd["label"] = "N-TPD"

ori = pd.concat([s_aptb, s_cdptb, s_itb, s_n_tpd], axis=0)
ori.dropna()
ori.reset_index(drop=True, inplace=True)


def result_row(row):
    if row.name in ids_train[y_train == "APTB"]:
        return "Train APTB"
    if row.name in ids_train[y_train == "CDPTB"]:
        return "Train CDPTB"
    if row.name in ids_train[y_train == "ITB"]:
        return "Train ITB"
    if row.name in ids_train[y_train == "N-TPD"]:
        return "Train N-TPD"
    if row.name in ids_test[(y_test == "APTB") & (y_pred == "APTB")]:
        return "Test APTB -> APTB"
    if row.name in ids_test[(y_test == "APTB") & (y_pred == "CDPTB")]:
        return "Test APTB -> CDPTB"
    if row.name in ids_test[(y_test == "APTB") & (y_pred == "ITB")]:
        return "Test APTB -> ITB"
    if row.name in ids_test[(y_test == "APTB") & (y_pred == "N-TPD")]:
        return "Test APTB -> N-TPD"
    if row.name in ids_test[(y_test == "CDPTB") & (y_pred == "APTB")]:
        return "Test CDPTB -> APTB"
    if row.name in ids_test[(y_test == "CDPTB") & (y_pred == "CDPTB")]:
        return "Test CDPTB -> CDPTB"
    if row.name in ids_test[(y_test == "CDPTB") & (y_pred == "ITB")]:
        return "Test CDPTB -> ITB"
    if row.name in ids_test[(y_test == "CDPTB") & (y_pred == "N-TPD")]:
        return "Test CDPTB -> N-TPD"
    if row.name in ids_test[(y_test == "ITB") & (y_pred == "APTB")]:
        return "Test ITB -> APTB"
    if row.name in ids_test[(y_test == "ITB") & (y_pred == "CDPTB")]:
        return "Test ITB -> CDPTB"
    if row.name in ids_test[(y_test == "ITB") & (y_pred == "ITB")]:
        return "Test ITB -> ITB"
    if row.name in ids_test[(y_test == "ITB") & (y_pred == "N-TPD")]:
        return "Test ITB -> N-TPD"
    if row.name in ids_test[(y_test == "N-TPD") & (y_pred == "APTB")]:
        return "Test N-TPD -> APTB"
    if row.name in ids_test[(y_test == "N-TPD") & (y_pred == "CDPTB")]:
        return "Test N-TPD -> CDPTB"
    if row.name in ids_test[(y_test == "N-TPD") & (y_pred == "ITB")]:
        return "Test N-TPD -> ITB"
    if row.name in ids_test[(y_test == "N-TPD") & (y_pred == "N-TPD")]:
        return "Test N-TPD -> N-TPD"


ori["result"] = ori.apply(lambda row: result_row(row), axis=1)

ori["prob_APTB"] = "-"
ori["prob_CDPTB"] = "-"
ori["prob_ITB"] = "-"
ori["prob_N-TPD"] = "-"

for idx, test_id in enumerate(ids_test):
    ori.loc[test_id, "prob_APTB"] = y_proba[idx][0]
    ori.loc[test_id, "prob_CDPTB"] = y_proba[idx][1]
    ori.loc[test_id, "prob_ITB"] = y_proba[idx][2]
    ori.loc[test_id, "prob_N-TPD"] = y_proba[idx][3]

ori.to_csv(f"data/{base}/result.csv", index=False)

# 11. Calculate the metrics and save them to csv file.
accuracy = accuracy_score(y_test, y_pred)

precisions = precision_score(y_test, y_pred, average=None)
recalls = recall_score(y_test, y_pred, average=None)
f1s = f1_score(y_test, y_pred, average=None)

precision_macro = precision_score(y_test, y_pred, average="macro")
recall_macro = recall_score(y_test, y_pred, average="macro")
f1_macro = f1_score(y_test, y_pred, average="macro")

cm = confusion_matrix(y_test, y_pred)
specificity = []
sensitivity = []
for i in range(len(cm)):
    # True negatives
    tn = sum(
        [cm[j][k] for j in range(len(cm)) if j != i for k in range(len(cm)) if k != i]
    )
    # False positives
    fp = sum(
        [cm[j][k] for j in range(len(cm)) if j != i for k in range(len(cm)) if k == i]
    )
    # False negatives
    fn = sum(
        [cm[j][k] for j in range(len(cm)) if j == i for k in range(len(cm)) if k != i]
    )
    # True positives
    tp = sum(
        [cm[j][k] for j in range(len(cm)) if j == i for k in range(len(cm)) if k == i]
    )

    specificity.append(tn / (tn + fp))
    sensitivity.append(tp / (tp + fn))

# Calculate average specificity and sensitivity
avg_specificity = sum(specificity) / len(specificity)
avg_sensitivity = sum(sensitivity) / len(sensitivity)

metrics = pd.DataFrame(
    {
        "Metrics": ["APTB", "CDPTB", "ITB", "N-TPD", "Average"],
        "Accuracy": ["-", "-", "-", "-", accuracy],
        "Precision": [
            *precisions,
            precision_macro,
        ],
        "Recall": [*recalls, recall_macro],
        "F1-Score": [*f1s, f1_macro],
        "Specificity": [*specificity, avg_specificity],
        "Sensitivity": [*sensitivity, avg_sensitivity],
    }
)

metrics.to_csv(f"data/{base}/metrics.csv", index=False)

# 12. Plot the confusion matrix.
label_names = ["APTB", "CDPTB", "ITB", "N-TPD"]
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.savefig(f"data/{base}/cm", dpi=300)
plt.close()

# 13. Plot the ROC curve.
plt.figure(figsize=(10, 6))
colors = ["r", "g", "b", "y"]

for i in range(len(colors)):
    # Compute ROC curve and ROC area for each class
    y_true_i = [v == label_names[i] for v in y_test]
    fpr, tpr, _ = roc_curve(y_true_i, y_proba[:, i])
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.plot(
        fpr,
        tpr,
        color=colors[i],
        lw=2,
        label="{}: AUC = {:.2f}".format(label_names[i], roc_auc),
    )

# Plot random guessing line
plt.plot([0, 1], [0, 1], color="grey", linestyle="--", label="Random Guessing")

# Set plot configurations
plt.xlim([-0.01, 1.01])
plt.ylim([-0.01, 1.01])
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("Receiver Operating Characteristic (ROC) Curves")
plt.legend(loc="lower right")
plt.grid(True)
plt.savefig(f"data/{base}/roc", dpi=300)
plt.close()

# 14. Plot the PR curve.
for idx, label in enumerate(best_rf.classes_):
    y_score = y_proba[:, idx]
    precision, recall, _ = precision_recall_curve(y_test, y_score, pos_label=label)
    plt.plot(recall, precision, label=label)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend(loc="lower left")
plt.savefig(f"data/{base}/prc", dpi=300)
plt.close()

# 15. Plot the feature importance.
importances = best_rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = X_train.columns
top_k = 10
plt.figure(figsize=(10, 6))
plt.title("Feature Importances")
plt.bar(range(top_k), importances[indices][:top_k], align="center")
plt.xticks(range(top_k), feature_names[indices][:top_k], rotation=90)
plt.xlabel("Features")
plt.ylabel("Importance")
plt.tight_layout()
plt.savefig(f"data/{base}/feature-importances", dpi=300)
plt.close()

# 16. Plot the decision tree.
# estimator = best_rf.estimators_[0]
# plt.figure(figsize=(10, 6))
# tree.plot_tree(
#     estimator,
#     filled=True,
#     feature_names=feature_names.to_list(),
#     class_names=label_names,
# )
# plt.savefig(f"data/{base}/estimator_0", dpi=300)
# plt.close()

# 17. Plot the SHAP values.
explainer = shap.TreeExplainer(best_rf)
shap_values = explainer.shap_values(X_test_std)
shap.initjs()
shap.summary_plot(shap_values[1], X_test_std, show=False)
plt.savefig(f"data/{base}/shap", dpi=300)
plt.close()

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


# 18. Plot the DCA curve (decision curve analysis).
def calculate_net_benefit(y_true, y_pred_proba, thresholds, is_all=False):
    net_benefit = np.zeros(thresholds.shape[0])

    for i, t in enumerate(thresholds):
        if is_all:
            y_pred = y_pred_proba >= 0
        else:
            y_pred = y_pred_proba >= t
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        net_benefit[i] = (tp - fp * (t / (1 - t))) / (tp + fp + tn + fn)

    return net_benefit


classes = best_rf.classes_
thresholds = np.linspace(0, 0.8, 100)

for class_index, class_label in enumerate(classes):
    plt.plot(
        thresholds, np.zeros(thresholds.shape[0]), label="Treat None"
    )

    all = calculate_net_benefit(
        y_test == class_label, y_proba[:, class_index], thresholds, is_all=True
    )
    all_pos = thresholds[all > 0]
    all_val = all[all > 0]
    plt.plot(all_pos, all_val, label="Treat All")

    curve = calculate_net_benefit(
        y_test == class_label, y_proba[:, class_index], thresholds, is_all=False
    )
    plt.plot(thresholds, curve, label="Prediction Model")

    plt.xlabel("Probability Threshold")
    plt.ylabel("Net Benefit")
    plt.title("Decision Curve Analysis for Class: " + class_label)
    plt.legend()
    plt.grid(True)
    ax = plt.gca()
    ax.xaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0, decimals=None))
    plt.savefig(f"data/{base}/dca-{class_label}", dpi=300)
    plt.close()
