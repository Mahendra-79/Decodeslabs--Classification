import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# ==================================================
# 1. LOAD DATASET
# ==================================================

iris = load_iris()

X = iris.data
y = iris.target

print("\nDataset Shape")
print("Features:", X.shape)
print("Target:", y.shape)

# ==================================================
# 2. CREATE DATAFRAME
# ==================================================

df = pd.DataFrame(X, columns=iris.feature_names)
df["species"] = y

print("\nFirst Five Rows")
print(df.head())

# ==================================================
# 3. DATASET INFORMATION
# ==================================================

print("\nDataset Information")
print(df.info())

print("\nStatistical Summary")
print(df.describe())

# ==================================================
# 4. CHECK MISSING VALUES
# ==================================================

print("\nMissing Values")
print(df.isnull().sum())

# ==================================================
# 5. DATA VISUALIZATION
# ==================================================

print("\nGenerating Pair Plot...")

sns.pairplot(df, hue="species")
plt.show()

# ==================================================
# 6. CORRELATION HEATMAP
# ==================================================

plt.figure(figsize=(8,6))

sns.heatmap(
    df.corr(),
    annot=True,
    cmap="coolwarm"
)

plt.title("Correlation Heatmap")
plt.show()

# ==================================================
# 7. TRAIN TEST SPLIT
# ==================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\nTraining Samples:", len(X_train))
print("Testing Samples:", len(X_test))

# ==================================================
# 8. FEATURE SCALING
# ==================================================

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ==================================================
# 9. TRAIN MULTIPLE MODELS
# ==================================================

models = {
    "KNN": KNeighborsClassifier(n_neighbors=3),

    "Decision Tree": DecisionTreeClassifier(
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )
}

results = {}

print("\nTraining Models...\n")

for name, model in models.items():

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    results[name] = accuracy

    print(f"{name} Accuracy: {accuracy:.4f}")

# ==================================================
# 10. BEST MODEL
# ==================================================

best_model_name = max(results, key=results.get)

print("\nBest Model:", best_model_name)

best_model = models[best_model_name]

# ==================================================
# 11. PREDICTIONS
# ==================================================

y_pred = best_model.predict(X_test)

# ==================================================
# 12. ACCURACY
# ==================================================

accuracy = accuracy_score(y_test, y_pred)

print("\nFinal Accuracy")
print(accuracy)

# ==================================================
# 13. CLASSIFICATION REPORT
# ==================================================

print("\nClassification Report")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)

# ==================================================
# 14. CONFUSION MATRIX
# ==================================================

cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(6,5))

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues"
)

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

plt.show()

# ==================================================
# 15. MODEL COMPARISON GRAPH
# ==================================================

plt.figure(figsize=(8,5))

plt.bar(
    results.keys(),
    results.values()
)

plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")

plt.show()

# ==================================================
# 16. FEATURE IMPORTANCE
# ==================================================

rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

rf.fit(X_train, y_train)

importance = rf.feature_importances_

feature_df = pd.DataFrame({
    "Feature": iris.feature_names,
    "Importance": importance
})

print("\nFeature Importance")
print(feature_df)

plt.figure(figsize=(8,5))

sns.barplot(
    x="Importance",
    y="Feature",
    data=feature_df
)

plt.title("Feature Importance")
plt.show()

# ==================================================
# 17. NEW FLOWER PREDICTION
# ==================================================

print("\nNew Flower Prediction")

sample = [[5.1, 3.5, 1.4, 0.2]]

sample_scaled = scaler.transform(sample)

prediction = best_model.predict(sample_scaled)

print(
    "Predicted Flower:",
    iris.target_names[prediction][0]
)

# ==================================================
# 18. PREDICTION PROBABILITY
# ==================================================

if hasattr(best_model, "predict_proba"):

    probabilities = best_model.predict_proba(
        sample_scaled
    )

    print("\nPrediction Probabilities")

    for flower, prob in zip(
        iris.target_names,
        probabilities[0]
    ):
        print(
            f"{flower}: {prob:.4f}"
        )

print("\nProject Completed Successfully")