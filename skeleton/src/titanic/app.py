"""Functions for creating the Flask application."""


from flask import Flask, Response, request
import joblib


class Scorer(Flask):
    """Flask app for scoring predictions using a given model."""

    def __init__(self, *args, model_path, **kwargs):
        super().__init__(*args, **kwargs)

        self.add_url_rule("/ping", view_func=self.ping)

        self._model = joblib.load(model_path)

    def ping(self):
        """Heartbeat endpoint."""
        return "pong", 200
