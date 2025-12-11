import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
def prepare_kepler_data(df: pd.DataFrame):
    df = df[df["koi_disposition"].isin(["CONFIRMED", "FALSE POSITIVE"])].copy() # filters the dataset and only keeps real and false positives
    df["label"] = (df["koi_disposition"] == "CONFIRMED").astype(int) # machine learning needs binary values, so this makes confirmed 1 and false positive 0

    drop_cols = ["koi_disposition", "kepid", "kepoi_name", "kepler_name"] # Drops non-feature columns that we dont need
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=col)
    df = df.dropna()  # deletes the rows with missing feature data that we need
    X = df.drop(columns=["label"])    # Split features/labels
    y = df["label"]

    return X, y

def make_train_test_split(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(  #80% training data 20% test data
        X, y,
        test_size=test_size,
        random_state=random_state, #keeps results reproducable for q1 2 and 3
        stratify=y # keeps class balance equal
    )
    scaler = StandardScaler() # esentially splits train and test variables, standardises them so variance is 0 mean is 1 for ml and NN
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test, scaler

