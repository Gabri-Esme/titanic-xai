# Predict survival using the trained model
from joblib import load
import pandas as pd 

# -------------------------------
# 1. Load processed test data and trained model
# -------------------------------
X_test = pd.read_csv("data/processed/processed_test.csv")  # Load processed data
model = load("results/model_output/gradient_boost_model.joblib")  # Load the trained model

if 'PassengerId' in X_test.columns:
    passenger_ids = X_test['PassengerId']
    X_test.drop(columns=['PassengerId'], inplace=True)

# -------------------------------
# 2. Make predictions
# -------------------------------
predictions = model.predict(X_test)

# -------------------------------
# 3. Add 'Survived' column and reattach 'PassengerID'
# -------------------------------
X_test['Survived'] = predictions  # 0 = did not survive, 1 = survived
X_test['PassengerId'] = passenger_ids.values

# -------------------------------
# 4. Optional: reorder columns
# -------------------------------
# Keep PassengerId first, Survived second, drop everything else
submission = X_test[['PassengerId', 'Survived']]

# -------------------------------
# 8. Save submission CSV
# -------------------------------
submission.to_csv("results/model_output/submission.csv", index=False)
print("Kaggle submission saved as submission.csv")