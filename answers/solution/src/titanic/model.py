# -*- coding: utf-8 -*-

"""Classes for creating and persisting (fitted) ML models."""

import joblib

from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline

from .features import PreprocessFeatures


class TitanicModel(BaseEstimator):
    def __init__(self, n_estimators=200):
        super().__init__()
        self.n_estimators = n_estimators

    def _build_pipeline(self):
        return Pipeline(
            steps=[
                ("preprocess", PreprocessFeatures()),
                ("model", RandomForestClassifier(n_estimators=self.n_estimators)),
            ]
        )

    def fit(self, X, y=None, **fit_params):
        estimator = self._build_pipeline()
        self.estimator_ = estimator.fit(X, y=y)
        return self

    def predict(self, X, **predict_params):
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict(X)
