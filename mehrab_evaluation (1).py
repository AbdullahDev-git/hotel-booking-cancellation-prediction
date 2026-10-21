# %% [markdown]
# # Part 2 - Mehrab: Evaluation, results table and test set
# Needs `tree_best_params.json` (written by Abdullah's section). The test set is used only once, at the end.

# %%
from part1_pipeline import *
import json
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate
from sklearn.metrics import classification_report, ConfusionMatrixDisplay, RocCurveDisplay

X_train, X_test, y_train, y_test = load_split()

# %% [markdown]
# ## 1. Cross-validation of all models (train set only, 5-fold stratified)

# %%
with open("tree_best_params.json") as f:
    TREE_DEPTH = json.load(f)["max_depth"]   # tuned by Abdullah

CANDIDATES = {
    "Dummy (most frequent)": DummyClassifier(strategy="most_frequent"),
    "Naive Bayes": GaussianNB(var_smoothing=1e-3),
    "Decision tree": DecisionTreeClassifier(max_depth=TREE_DEPTH, class_weight="balanced", random_state=RANDOM_STATE),
}
SCORING = ["accuracy", "f1", "recall", "precision", "roc_auc"]

means, stds = {}, {}
for name, est in CANDIDATES.items():
    r = cross_validate(make_model_pipeline(est), X_train, y_train, cv=make_cv(), scoring=SCORING)
    means[name] = {m: r[f"test_{m}"].mean() for m in SCORING}
    stds[name] = {m: r[f"test_{m}"].std() for m in SCORING}

# %% [markdown]
# ## 2. Results table and chart

# %%
table = pd.DataFrame({name: {m: f"{means[name][m]:.3f} ± {stds[name][m]:.3f}" for m in SCORING}
                      for name in CANDIDATES}).T
table.columns = ["Accuracy", "F1", "Recall", "Precision", "ROC-AUC"]
table

# %%
mean_df = pd.DataFrame(means).T[SCORING]
std_df = pd.DataFrame(stds).T[SCORING]
ax = mean_df.plot(kind="bar", yerr=std_df, capsize=4, rot=0, figsize=(11, 5), colormap="viridis")
ax.set_ylim(0, 1)
ax.set_ylabel("Score (mean ± std over 5 folds)")
ax.set_title("5-fold stratified CV on the train set")
ax.legend(loc="lower right", ncol=5)
plt.tight_layout(); plt.show()

# %% [markdown]
# ## 3. Final models and one-time test-set check
# Chosen from the CV table above using F1 and ROC-AUC: the best Naive Bayes and the tuned decision tree. The test set is used once here, with no further tuning.

# %%
FINAL = {
    "Naive Bayes": GaussianNB(var_smoothing=1e-3),
    "Decision tree": DecisionTreeClassifier(max_depth=TREE_DEPTH, class_weight="balanced", random_state=RANDOM_STATE),
}
final_models = {name: make_model_pipeline(est).fit(X_train, y_train) for name, est in FINAL.items()}

LABELS = ["Not canceled", "Canceled"]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for ax, (name, model) in zip(axes, final_models.items()):
    y_pred = model.predict(X_test)
    print(f"===== {name} (test set) =====")
    print(classification_report(y_test, y_pred, target_names=LABELS))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=LABELS, cmap="Blues", ax=ax)
    ax.set_title(f"{name}: confusion matrix (test)")
plt.tight_layout(); plt.show()

# %%
fig, ax = plt.subplots(figsize=(6.5, 6))
for name, model in final_models.items():
    RocCurveDisplay.from_estimator(model, X_test, y_test, name=name, ax=ax)
ax.plot([0, 1], [0, 1], "k--", label="Random guess")
ax.set_title("ROC curves on the test set")
ax.legend(loc="lower right")
plt.tight_layout(); plt.show()
