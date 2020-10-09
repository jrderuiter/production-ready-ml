"""Classes for creating and persisting (fitted) ML models."""

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

from .features import PreprocessFeatures


class TitanicModel(BaseEstimator):
    """
    A RandomForest-based classifier that predicts survival on the Titanic.

    Parameters
    ----------
    n_estimators
        Number of trees to use when training the model.
    """

    def __init__(self, n_estimators: int = 200):
        super().__init__()
        self.n_estimators = n_estimators

    def _build_pipeline(self):
        return Pipeline(
            steps=[
                ("preprocess", PreprocessFeatures()),
                ("model", RandomForestClassifier(n_estimators=self.n_estimators)),
            ]
        )

    # pylint: disable=missing-function-docstring,unused-argument
    def fit(self, X, y=None, **fit_params):
        estimator = self._build_pipeline()
        self.estimator_ = estimator.fit(X, y=y)
        return self

    # pylint: disable=missing-function-docstring,unused-argument
    def predict(self, X, **predict_params):
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict(X)
