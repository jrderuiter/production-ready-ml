import pandas as pd


def preprocess(dataset):
    """Preprocesses the given dataset to require our features."""
    return (
        dataset
        .pipe(select_features)
        .pipe(impute_missing_values)
        .pipe(encode_categorical)
    )


def select_features(df):
    """Selects the features we're interested in."""
    return df[["Pclass", "Sex"]]


def impute_missing_values(df):
    """Imputes missing values by filling in the most frequently occurring value."""
    most_frequent_values = df.mode().iloc[0]
    return df.fillna(most_frequent_values)


def encode_categorical(df):
    """One-hot encodes any categorical features."""
    return pd.get_dummies(df, drop_first=True)
