from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from titanic.features import PreprocessFeatures

# Example showing how to use external files in fixtures:

# @pytest.fixture()
# def example_dataset_path():
#     data_dir = Path(__file__).parent / "data"
#     return data_dir / "example.csv"


# @pytest.fixture()
# def example_dataset_ext(example_dataset_path):
#     """Fixture that loads an external dataset."""
#     return pd.read_csv(example_dataset_path)


# def test_ext(example_dataset_ext):
#     print(example_dataset_ext)


class TestPreprocessFeatures:
    @pytest.fixture()
    def example_dataset(self):
        """Example dataset containing missing values."""
        return pd.DataFrame(
            {"Pclass": [1, 1, 2, 3], "Sex": ["male", "female", "female", np.nan]}
        )

    @pytest.fixture()
    def expected_result(self):
        return np.array([[1.0, 1.0], [1.0, 0.0], [2.0, 0.0], [3.0, 0.0]])

    def test_example(self, example_dataset, expected_result):
        """Tests basic transform on an example dataset."""

        transformer = PreprocessFeatures()
        result = transformer.fit_transform(example_dataset)

        assert (result == expected_result).all()

    def test_extra_column(self, example_dataset, expected_result):
        """Tests if an extra column does not change our results."""

        example_dataset = example_dataset.assign(extra=1)

        transformer = PreprocessFeatures()
        result = transformer.fit_transform(example_dataset)

        assert (result == expected_result).all()

    def test_missing_column(self, example_dataset):
        """Tests if missing column raises a ValueError."""

        example_dataset = example_dataset.drop(["Sex"], axis=1)
        transformer = PreprocessFeatures()

        with pytest.raises(ValueError):
            transformer.fit_transform(example_dataset)

    def test_no_fit(self, example_dataset):
        """Tests what happens if we forget to fit."""

        transformer = PreprocessFeatures()

        with pytest.raises(NotFittedError):
            transformer.transform(example_dataset)
