# -*- coding: utf-8 -*-

"""Classes for creating and persisting (fitted) ML models."""

import joblib

from sklearn import metrics
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import cross_validate
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


class TitanicModel(BaseEstimator):

    def __init__(self, n_estimators=200):
        super().__init__()
        self.n_estimators = n_estimators

    def fit(self, X, y=None, **fit_params):
        """
        Fits model on the given dataset.

        Parameters
        ----------
        X : pd.Dataframe
            Dataframe containing training data (features only, no response).
        y : Union[pd.Series, np.ndarray]
            Pandas series (or numpy array) containing the response values for the
            given training dataset.

        Returns
        -------
        Model
            Returns the model itself, after fitting.
        """
        estimator = RandomForestClassifier(n_estimators=self.n_estimators)
        self.estimator_ = estimator.fit(X, y=y)
        return self

    def predict(self, X, **predict_params):
        check_is_fitted(self, ["estimator_"])
        return self.estimator_.predict(X)
