

# %%
from part1_pipeline import *
import jsongi
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import cross_validate

X_train, X_test, y_train, y_test = load_split()



# %%
tree_pipe = make_model_pipeline(DecisionTreeClassifier(class_weight="balanced", random_state=RANDOM_STATE))
tree_pipe.fit(X_train, y_train)
full_tree = tree_pipe.named_steps["model"]
print("Full-depth tree | train accuracy: %.3f | leaves: %d | depth: %d"
      % (tree_pipe.score(X_train, y_train), full_tree.get_n_leaves(), full_tree.get_depth()))


# %%
DEPTHS = [3, 5, 8, 12, None]
rows = []
for d in DEPTHS:
    pipe = make_model_pipeline(DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=RANDOM_STATE))
    r = cross_validate(pipe, X_train, y_train, cv=make_cv(), scoring=["f1", "roc_auc"], return_train_score=True)
    rows.append({"max_depth": "None" if d is None else d,
                 "train F1": r["train_f1"].mean(),
                 "CV F1": r["test_f1"].mean(),
                 "CV F1 std": r["test_f1"].std(),
                 "CV ROC-AUC": r["test_roc_auc"].mean()})
depth_df = pd.DataFrame(rows)
depth_df.round(3)

# %%
x = np.arange(len(DEPTHS))
fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(x, depth_df["train F1"], "o-", label="Train F1")
ax.errorbar(x, depth_df["CV F1"], yerr=depth_df["CV F1 std"], fmt="s-", capsize=4, label="CV F1")
ax.set_xticks(x, depth_df["max_depth"].astype(str))
ax.set_xlabel("max_depth"); ax.set_ylabel("F1")
ax.set_title("Depth vs train and CV F1 (overfitting)")
ax.legend(); plt.tight_layout(); plt.show()

best_depth = DEPTHS[int(depth_df["CV F1"].values.argmax())]
print("Best max_depth by CV F1:", best_depth)
with open("tree_best_params.json", "w") as f:
    json.dump({"max_depth": best_depth}, f)



# %%
best_tree = make_model_pipeline(DecisionTreeClassifier(max_depth=best_depth, class_weight="balanced", random_state=RANDOM_STATE))
best_tree.fit(X_train, y_train)
names = selected_feature_names(best_tree.named_steps["preprocess"])
clf = best_tree.named_steps["model"]
print(f"Root split: {names[clf.tree_.feature[0]]} <= {clf.tree_.threshold[0]:.3f}  (numeric features are scaled, so the threshold is in standard deviations)")

# %%
fig, ax = plt.subplots(figsize=(22, 9))
plot_tree(clf, max_depth=3, feature_names=list(names), class_names=["Not canceled", "Canceled"],
          filled=True, fontsize=9, ax=ax)
plt.title("Decision tree (top 3 levels)")
plt.tight_layout(); plt.show()

print(export_text(clf, max_depth=3, feature_names=list(names)))

# %%
imp = pd.Series(clf.feature_importances_, index=names).sort_values(ascending=False)
imp.head(15).iloc[::-1].plot(kind="barh", figsize=(8, 6), color="teal")
plt.title("Top 15 feature importances (tuned tree)")
plt.xlabel("Importance"); plt.tight_layout(); plt.show()


# %%
Xt = pd.DataFrame(best_tree.named_steps["preprocess"].transform(X_train), columns=names)
corr = Xt.corr()

plt.figure(figsize=(12, 10))
sns.heatmap(corr, cmap="coolwarm", vmin=-1, vmax=1, xticklabels=True, yticklabels=True)
plt.title("Correlation between the selected features (train set)")
plt.tight_layout(); plt.show()

upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
pairs = upper.stack()
strong = pairs[pairs.abs() > 0.8].sort_values(key=lambda s: s.abs(), ascending=False)
print("Feature pairs with |correlation| > 0.8:")
print(strong.round(3).to_string() if len(strong) else "None")