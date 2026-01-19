import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_recall_curve

from preprocessing import load_and_preprocess

os.makedirs("output/evaluation", exist_ok=True)

model = joblib.load("models/best_model.pkl")
X, y, _ = load_and_preprocess()

probs = model.predict_proba(X)[:, 1]
preds = (probs >= 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y, preds)
sns.heatmap(cm, annot=True, fmt="d")
plt.title("Confusion Matrix")
plt.savefig("output/evaluation/confusion_matrix.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y, probs)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f"AUC={roc_auc:.2f}")
plt.plot([0, 1], [0, 1], "--")
plt.legend()
plt.title("ROC Curve")
plt.savefig("output/evaluation/roc_curve.png")
plt.close()

# Precision Recall
precision, recall, _ = precision_recall_curve(y, probs)
plt.plot(recall, precision)
plt.title("Precision-Recall Curve")
plt.savefig("output/evaluation/pr_curve.png")
plt.close()
