import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def prepare_kepler_data(df: pd.DataFrame):

    #include "CANDIDATE" so dataset is not empty after filtering
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE", "CANDIDATE"])].copy()
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int)  # CONFIRMED = 1, everything else = 0
    drop_cols = ["koi_disposition", "kepid", "kepoi_name", "kepler_name"]  # Drops non-feature columns
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)
    numeric_cols = df.select_dtypes(include=["float", "int"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    X = df.drop(columns=["label"])   # Split features/labels
    y = df["label"]
    return X, y
def make_train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(  # 80% train, 20% test
        X, y,
        test_size=test_size,
        random_state=random_state,  # keeps results reproducible
        stratify=y  # keeps class balance equal
    )
    scaler = StandardScaler()  # standardises data (mean 0, variance 1)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
