## Titanic Survival Prediction with Explainable AI

# Overview
This project explores the Titanic dataset from Kaggle to predict passenger survival. The focus is on understanding which features influence survival using explainable AI (XAI) techniques, instead of simply maximising accuracy. The dataset is observational, so all conclusions are about patterns in the data rather than causal effects.

---

# Workflow

1. Data Exploration
    - Investigated distributions, missing values, and feature relationships.
    - Visualised correlations and basic statistics to inform preprocessing.
2. Feature Engineering 
    - Extracted titles from passenger names (e.g., Mr, Mrs, Master).
    - Derived CabinLetter from the first letter of the cabin.
    - Filled missing values for Age, Fare, and Embarked.
    - One-hot encoded categorical variables.
    - Created a clean feature set of ~32 features for modeling.
3. Baseline Models
    - Trained three baseline models:
        - Random Forest Classifier
        - LASSO Logistic Regression
        - Gradient Boosting Classifier
    - Gradient Boosting achieved the highest accuracy (0.8380 on training data).
4. Feature Importance
    - Computed using:
        - Random Forest feature importance
        - Permutation importance
        - LASSO coefficients
        - SHAP values (TreeExplainer)
    - Key features: Fare, Age, Sex, Title_Mr, Pclass, SibSp.
5. Explainable AI
    - Applied SHAP TreeExplainer to the trained Gradient Boosting model.
    - Force and summary plots highlight which features push predictions toward survival or non-survival.
    - Observed patterns align with historical knowledge (“women and children first”, higher class, and higher fare increase survival probability).

---


# Notes
- This is an observational analysis; SHAP explanations show influential features, not causation.
- The project is designed as a portfolio showcase of feature engineering, model building, and explainable AI techniques.
- All preprocessing and model pipelines are reproducible and reusable for test set predictions.

---

# Folder Structure  

titanic-causal-xai/  
│
├── README.md  
├── requirements.txt  
├── data/  
│   ├── raw/             # empty folder for original Kaggle dataset (not committed)  
│   ├── processed/       # empty folder for processed dataset  
│  
├── notebooks/  
│   ├── 01_exploration.ipynb  
│   ├── 02_preprocessing.ipynb  
│   ├── 03_feature_importance.ipynb  
│   ├── 04_baseline_models.ipynb  
│   └── 05_model_explainability.ipynb  
│  
├── src/  
│   ├── preprocessing_and_feature_engineering.py  
│   ├── modeling.py  
│   └── prediction.py  
│  
├── results/  
│   ├── figures/         # plots, DAG diagrams, SHAP plots (optional and not commited)  
│   └── model_outputs/   # predictions, metrics, serialized models  
└── .gitignore  

---

# Submission
- Predictions were generated for Kaggle test set using the Gradient Boosting model.
- PassengerId was preserved for submission integrity.
- Kaggle score: 0.7703.
