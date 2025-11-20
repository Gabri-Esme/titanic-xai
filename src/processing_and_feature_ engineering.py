import pandas as pd

def preprocess_titanic(df, select_features=True):
    """
    Preprocess Titanic dataset for modeling.

    Parameters:
    -----------
    df : pd.DataFrame
        Raw Titanic dataset (train or test)
    select_features : bool
        If True, select only the final features used for modeling

    Returns:
    --------
    df_processed : pd.DataFrame
        Preprocessed DataFrame ready for modeling
    """

    # -------------------------------
    # 0. Keep PassengerId if needed
    # -------------------------------
    if 'PassengerId' in df.columns:
        passenger_ids = df['PassengerId']  # store IDs before dropping

    # -------------------------------
    # 1. Drop irrelevant columns
    # -------------------------------
    # Ticket is usually not informative for survival prediction
    if 'Ticket' in df.columns:
        df = df.drop(columns=['Ticket'])
    
    # -------------------------------
    # 2. Handle missing values
    # -------------------------------
    if 'Age' in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())

    if 'Fare' in df.columns:  # especially for test set
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    if 'Embarked' in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if 'Cabin' in df.columns:
        df["CabinLetter"] = df["Cabin"].fillna("U").str[0]
        df = df.drop(columns=['Cabin'])

    # -------------------------------
    # 3. Extract Title from Name
    # -------------------------------
    if 'Name' in df.columns:
        df['Title'] = df['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
        df = df.drop(columns=['Name'])

    # -------------------------------
    # 4. Encode categorical variables
    # -------------------------------
    cat_cols = [col for col in ['Sex', 'Embarked', 'CabinLetter', 'Title'] if col in df.columns]
    if cat_cols:
        df = pd.get_dummies(df, columns=cat_cols, drop_first=True)

    # -------------------------------
    # 5. Select relevant features
    # -------------------------------
    if select_features:
        features = ['Fare', 'Age', 'Sex_male', 'Pclass', 'SibSp']
        if 'Survived' in df.columns:
            features.append('Survived')  # keep target if present
        # Add missing columns (important for test set)
        for col in features:
            if col not in df.columns:
                df[col] = 0
        df = df[features]

    # -------------------------------
    # 6. Reattach PassengerId if needed
    # -------------------------------
    df['PassengerId'] = passenger_ids.values

    return df

# Process datasets

train_df = pd.read_csv("data/raw/train.csv")
processed_train = preprocess_titanic(train_df)
processed_train.to_csv("data/processed/processed_train.csv", index=False)

test_df = pd.read_csv("data/raw/test.csv")
processed_test = preprocess_titanic(test_df)
processed_test.to_csv("data/processed/processed_test.csv", index=False)


