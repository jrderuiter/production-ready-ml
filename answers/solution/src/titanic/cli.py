import json
import logging
from pathlib import Path

import click
import joblib
import pandas as pd
from sklearn import metrics
from sklearn.model_selection import cross_validate

from titanic.app import Scorer
from titanic.model import TitanicModel

logging.basicConfig(
    level=logging.INFO, format="[%(asctime)-15s] %(name)s - %(levelname)s - %(message)s"
)


@click.group()
def cli():
    """CLI for our titanic model."""


@cli.command()
@click.option(
    "--input_path",
    type=click.Path(dir_okay=False, exists=True),
    required=True,
    help="Input dataset.",
)
@click.option(
    "--model_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Where to save the output model.",
)
@click.option("--n_estimators", type=int, default=200)
def train(input_path, model_path, n_estimators):
    """Trains a model on the given dataset."""

    logger = logging.getLogger(__name__)

    logger.info("Loading input dataset")
    train_dataset = pd.read_csv(input_path)

    X_train = train_dataset.drop(["Survived"], axis=1)
    y_train = train_dataset["Survived"]

    logger.info(f"Training model with n_estimators = {n_estimators}")
    model = TitanicModel(n_estimators=n_estimators)
    model.fit(X_train, y=y_train)

    logger.info(f"Writing output to {model_path}")
    model_dir = Path(model_path).parent
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)


@cli.command()
@click.option(
    "--input_path",
    type=click.Path(dir_okay=False, exists=True),
    required=True,
    help="Validation dataset.",
)
@click.option(
    "--model_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Trained model.",
)
@click.option(
    "--metrics_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Where to write output metrics.",
)
def evaluate(input_path, model_path, metrics_path):
    """Evaluates a trained model on the given dataset."""

    logger = logging.getLogger(__name__)

    logger.info("Loading input dataset")
    dataset = pd.read_csv(input_path)

    X_eval = dataset.drop("Survived", axis=1)
    y_eval = dataset["Survived"]

    logger.info("Loading model")
    model = joblib.load(model_path)

    logger.info("Calculating metrics")
    scorer = metrics.make_scorer(metrics.mean_squared_error)
    cv_results = cross_validate(model, X=X_eval, y=y_eval, scoring=scorer, cv=5)

    metric_values = {"mse": cv_results["test_score"].mean()}

    logger.info(f"Writing output to {metrics_path}")
    with open(metrics_path, "w") as file_:
        json.dump(metric_values, file_)


@cli.command()
@click.option(
    "--input_path",
    type=click.Path(dir_okay=False, exists=True),
    required=True,
    help="Prediction dataset.",
)
@click.option(
    "--model_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Trained model.",
)
@click.option(
    "--output_path",
    type=click.Path(dir_okay=False),
    required=True,
    help="Where to write output predictions.",
)
def predict(input_path, model_path, output_path):
    """Makes predictions for the given dataset."""

    logger = logging.getLogger(__name__)

    logger.info("Loading input dataset")
    X_pred = pd.read_csv(input_path)

    logger.info("Loading model")
    model = joblib.load(model_path)

    logger.info("Generating predictions")
    predictions = model.predict(X_pred)
    prediction_df = pd.DataFrame({"predictions": predictions})

    logger.info(f"Writing output to {output_path}")
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_df.to_csv(output_path, index=False)


@cli.command()
@click.argument(
    "model_path",
    type=click.Path(dir_okay=False, exists=True),
)
@click.option("--host", default="127.0.0.1")
@click.option("--port", default=5000)
@click.option("--debug/--no-debug", default=False)
def serve(model_path, host, port, debug):
    """Serves the given model in an API end point."""

    app = Scorer(__name__, model_path=model_path)
    app.run(host=host, port=port, debug=debug)
