# -------------------------------
# Gradient Boosting Classifier
# -------------------------------

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# 1. Load processed data
# -------------------------------
df = pd.read_csv("data/processed/processed_train.csv")  # Corrected path and quotes

# -------------------------------
# 2. Define features and target
# -------------------------------
cols_to_drop = ['Survived']
if 'PassengerId' in df.columns:
    cols_to_drop.append('PassengerId')

X = df.drop(columns=cols_to_drop)
y = df['Survived']

# -------------------------------
# 3. Split into train and test sets
# -------------------------------
# Using 80-20 split and fixed random_state for reproducibility
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# 4. Initialize Gradient Boosting model
# -------------------------------
# n_estimators: number of boosting stages
# learning_rate: step size for each stage
# max_depth: max depth of individual trees
gb = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=3,
    random_state=42
)

# -------------------------------
# 5. Train the model
# -------------------------------
gb.fit(X_train, y_train)

# -------------------------------
# 6. Make predictions
# -------------------------------
y_pred_gb = gb.predict(X_test)

# -------------------------------
# 7. Evaluate the model
# -------------------------------
print(f"GradientBoosting Accuracy: {accuracy_score(y_test, y_pred_gb):.4f}\n")

print("Classification Report:")
print(classification_report(y_test, y_pred_gb))

print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred_gb))

# -------------------------------
# Optional: Save the trained model
# -------------------------------
from joblib import dump
dump(gb, "results/model_output/gradient_boost_model.joblib")  # Save for later predictions or XAI
