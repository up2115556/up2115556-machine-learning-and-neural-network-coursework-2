import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def prepare_kepler_data(df: pd.DataFrame):
    # include "CANDIDATE" so dataset is not empty after filtering
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"])].copy()
    # machine learning needs binary values, so this makes confirmed 1 and everything else 0
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)
    # drops non-feature columns that we dont need
    drop_cols = ["koi_disposition", "kepid", "kepoi_name", "kepler_name", "koi_pdisposition", "koi_tce_delivname"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    # drops columns where every value is missing
    df = df.dropna(axis=1, how="all")
    # convert columns to numeric where possible
    df = df.convert_dtypes()
    # select numeric columns for imputation
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    # imputes missing numeric values using median
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median(numeric_only=True))
    # keep only numeric columns for the model
    df = df.select_dtypes(include=["float", "int"])
    # split features/labels
    X = df.drop(columns=["label"])
    y = df["label"]
    return X, y
def make_train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(  # 80% training data 20% test data
        X, y,
        test_size=test_size,
        random_state=random_state,  # keeps results reproducable for q1 2 and 3
        stratify=y  # keeps class balance equal
    )
    scaler = StandardScaler()  # standardises them so variance is 0 mean is 1 for ml and NN
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
