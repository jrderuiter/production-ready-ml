import pandas as pd

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.utils.validation import check_is_fitted


class PreprocessFeatures(BaseEstimator, TransformerMixin):

    def __init__(self):
        super().__init__()

    def _build_transformer(self):
        transform_categorical = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="error", drop="first"))
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("numerical", SimpleImputer(strategy="most_frequent"), ["Pclass"]),
                ("categorical", transform_categorical, ["Sex"])
            ],
            remainder="drop"
        )

        return transformer

    def fit(self, X, y=None):
        transformer = self._build_transformer()
        self.transformer_ = transformer.fit(X, y=y)
        return self

    def transform(self, X):
        check_is_fitted(self, ["transformer_"])
        return self.transformer_.transform(X)


def build_preprocessor():
    """Preprocesses the given dataset to require our features."""

    transform_categorical = Pipeline(
        steps=[
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="error", drop="first"))
        ]
    )

    transformer = ColumnTransformer(
        transformers=[
            ("numerical", SimpleImputer(strategy="most_frequent"), ["Pclass"]),
            ("categorical", transform_categorical, ["Sex"])
        ],
        remainder="drop"
    )

    return transformer


transformer.fit_transform(dataset)

    # return (
    #     dataset
    #     .pipe(select_features)
    #     .pipe(impute_missing_values)
    #     .pipe(encode_categorical)
    # )


# def select_features(df):
#     """Selects the features we're interested in."""
#     return df[["Pclass", "Sex"]]


# def impute_missing_values(df):
#     """Imputes missing values by filling in the most frequently occurring value."""
#     most_frequent_values = df.mode().iloc[0]
#     return df.fillna(most_frequent_values)


# def encode_categorical(df):
#     """One-hot encodes any categorical features."""
#     return pd.get_dummies(df, drop_first=True)
