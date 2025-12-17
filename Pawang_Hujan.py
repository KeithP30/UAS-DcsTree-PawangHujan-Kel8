import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt


rng = np.random.default_rng(1)  

num_samples = 150

df = pd.DataFrame({
    "temperature": rng.integers(17, 42, num_samples),     # suhu: 17–42
    "humidity": rng.integers(30, 95, num_samples),        # kelembapan: 30–95
    "pressure": rng.integers(995, 1025, num_samples),     # tekanan: 995–1025
    "wind_speed": rng.uniform(1.0, 6.0, num_samples),     # kecepatan angin: 1.0–6.0
})

def generate_label(row):
    if row["humidity"] < 55 and row["temperature"] > 28:
        return "Sunny"
    elif row["humidity"] > 75 and row["pressure"] < 1008:
        return "Rain"
    else:
        return "Cloudy"


df["label"] = df.apply(generate_label, axis=1)

print("Dataset Random 150 Baris:")
print(df)


X = df[["temperature", "humidity", "pressure", "wind_speed"]]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y
)

model = DecisionTreeClassifier(random_state=1, max_depth=3, min_samples_leaf=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
print("Confusion matrix:\n", cm)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=model.classes_, yticklabels=model.classes_)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.show()


feat_imp = pd.Series(model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(6,3))
sns.barplot(x=feat_imp.values, y=feat_imp.index)
plt.title("Feature Importance")
plt.tight_layout()
plt.show()

top2_features = feat_imp.index[:2]
plt.figure(figsize=(6,5))
sns.scatterplot(
    data=df,
    x=top2_features[0],
    y=top2_features[1],
    hue="label",
    style="label"
)
plt.xlabel(top2_features[0])
plt.ylabel(top2_features[1])
plt.title(f"Scatter Plot: {top2_features[0]} vs {top2_features[1]}")
plt.tight_layout()
plt.show()



plt.figure(figsize=(10,6))
plot_tree(model, feature_names=X.columns, class_names=model.classes_, filled=True, rounded=True)
plt.title("Decision Tree")
plt.show()



new_samples = pd.DataFrame([
    [31, 38, 1016, 1.7],
    [19, 89, 999, 5.8],
    [26, 62, 1011, 3.0],
], columns=["temperature", "humidity", "pressure", "wind_speed"])

print("\nNew samples:")
print(new_samples)
print("\nPredictions:", model.predict(new_samples))
print("Probabilities:\n", model.predict_proba(new_samples))
