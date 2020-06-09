"""Functions for creating the Flask application."""

import io

from flask import Flask, Response, request
import joblib
import pandas as pd


class Scorer(Flask):
    """Flask app for scoring predictions using a given model."""

    def __init__(self, *args, model_path, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_url_rule("/ping", view_func=self.ping)
        self.add_url_rule("/predict", view_func=self.predict, methods=["POST"])

        self._model = joblib.load(model_path)

    def ping(self):
        """Heartbeat endpoint."""
        return "pong", 200

    def predict(self):
        """Predict endpoint, which produces predictions for a given dataset."""

        data = pd.read_json(io.BytesIO(request.get_data()))

        y_pred = self._model.predict(data)
        y_pred_df = pd.DataFrame({"prediction": y_pred})

        return Response(y_pred_df.to_json(), content_type="application/json",)
