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
                ("encode", OneHotEncoder(handle_unknown="error", drop="first")),
            ]
        )

        transformer = ColumnTransformer(
            transformers=[
                ("numerical", SimpleImputer(strategy="most_frequent"), ["Pclass"]),
                ("categorical", transform_categorical, ["Sex"]),
            ],
            remainder="drop",
        )

        return transformer

    def fit(self, X, y=None):
        transformer = self._build_transformer()
        self.transformer_ = transformer.fit(X, y=y)
        return self

    def transform(self, X):
        check_is_fitted(self, ["transformer_"])
        return self.transformer_.transform(X)
