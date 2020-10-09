# Production-ready Machine Learning training

[![Open in Gitpod](https://gitpod.io/button/open-in-gitpod.svg)](https://gitpod.io/#https://github.com/jrderuiter/production-ready-ml)

This repository contains the materials for the Production-ready Machine Learning training.

## Use case

We have received an initial implementation of a machine learning model from one of our data scientists. They have asked us to convert their code into a well-documented and well-tested Python package, that can easily be distributed within the company.

Their full list of requests include:

* Converting this code base to a Semver versioned Python package.
* Improve the quality of the implementation and code
* Introducing tests to verify that the code works correctly.
* Creating a command line interface for training/evaluating/predicting.
* Implementing a Flask REST API that exposes a trained model and returns predictions.

Besides this, they would also like to have an easy way of running a linter (pylint) and code formatter (black) on their code. Ideally, this would also run as part of a CI/CD pipeline, as well as being easy to run locally.

## Exercises

The below exercises will help you implement these different wishes. For a starting point, see the code provided in the **skeleton** directory.

### 1 - Packaging

As a first step, let's start by creating a Python package. Typically this includes creating a setup.py file, which contains metadata on your package, telling Python how it should be installed.

1. Create a setup.py file that allows you to install your package. Ensure the setup.py file contains a proper package name, version and other metadata such as author name, dependencies, etc.
2. Verify that you can install your package using pip and that you can import and run code from your package using the Python REPL.
3. Build a source/wheel distribution of your package (which you could use for re-distributing your package). *Tip: See https://pypi.org/project/pep517/ for the command you need to build distributions using the pyproject.toml file.*

*Bonus*

4. Try using a tool like setuptools_scm to automatically version your package based on git tags.

*References*
* https://manikos.github.io/a-tour-on-python-packaging
* https://www.bernat.tech/pep-517-and-python-packaging
* Interesting discussion about using src directories:
    * https://hynek.me/articles/testing-packaging/
    * https://github.com/pypa/packaging.python.org/issues/320

### 2 - Switching to OO-based scikit-learn

Going over the code, we see that the pre-processing code in *features.py* makes heavy use of plain Python functions. Whilst this keeps things nice and simple, there are some issues with the current implementation.

* Can you imagine what issues we are referring to? *Tip: think of differences between train/predict datasets and how this may affect these functions.*
* How would you go about fixing these issues?

In this case, we decide to make things a bit more robust by implementing the preprocessing using scikit-learn transformers.

1. Implement the pre-processing using scikit-learn transformers. *Tip: You may be able to use some built-in transformers. Another tip: Look at the ColumnTransformer if you want to apply different transformations to different columns.*
2. Try running all the transformers together in a single pipeline to get a similar result as the original *preprocess* function.
3. Try including the model in the same pipeline, so that preprocessing + training/predicting can be done in one swoop.

### 3 - Setting up linting and code formatting

Now we have a basic package and implementation, it's a good idea to start enforcing some practices for maintaining good code quality.

1. Install pylint and see if you can run it on your code. Does it flag any issues in the code? If so, see if you can fix these issues. Alternatively, if you don't agree with pylint, see if you can disable certain warnings using a pylintrc file.
2. Install and run black on your code. Does black make any formatting changes to your code? Play with some examples and see what black does to make your code nicer (hopefully).
3. Create a Makefile with basic commands such as *pylint* and *black* which run pylint/black on your code base. This should allow the DS to easily run these checks using a command such as `make pylint`.

*Bonus*

4. Install and setup pre-commit to run pylint/black as [pre-commit](https://pre-commit.com/) hooks that run whenever you try to create a new git commit.
5. Add typehinting to your code base and check if you can verify whether your code is correct (at least as far as types are concerned) using a tool such as mypy.

### 4 - Introducing tests using pytest

Now we setup our code quality checks, we should probably also start implementing some tests for our code.

1. Install pytest and create a *tests/titanic* directory, in which we will start writing our tests.
2. Create a *test_features.py* file and implement some tests for our *PreprocessFeatures* class. See if you can use fixtures for sharing test data across tests.

*Bonus*

3. Design your test data fixtures to load data from external test files, rather than defining your test data inline.
4. Add a *test* command to your Makefile for running your tests.
5. Create tests for the model class.

### 5 - Creating a command line interface (using click)

To increase the usability of our package, we would like to implement the following command line scripts for our package:

* Train - trains our model on a given dataset, producing a model pkl file.
* Evaluate - loads a trained model from a pkl file and evaluates the model on a validation dataset.
* Predict - loads a trained model and produces predictions for a given prediction dataset.

Preferably, we would also like to add some logging output to give feedback to the user when running our command.

To do so, try the following:

1. Create a *cli* module in the *titanic* package, which will contain entrypoints for our command line interface.
2. Define *train*, *evaluate* and *predict* functions in this module and use [click](https://click.palletsprojects.com/en/7.x/) to convert these functions into a command line interface. *Tip: use command groups to group the three commands.*
3. Use entrypoints in setup.py (https://chriswarrick.com/blog/2014/09/15/python-apps-the-right-way-entry_points-and-scripts/) to automatically create a command line app when your package is installed.
4. Reinstall your package after adding the entrypoints and see if you can run your command line application.
5. Add logging to one (or all) of the commands. Don't forget to configure logging correctly to actually get your log output.

### 6 - Implementing a REST API using Flask

As a final step, we would like to add a small REST-based API for serving our model. The idea is that this REST API includes a `/predict` endpoint, which should accept a JSON dataset payload and return a JSON payload containing predictions.

1. Add an app module in your Python package that implements a [Flask](https://flask.palletsprojects.com/en/1.1.x/) API with a `ping` endpoint, which simply returns 'pong' when called. You can find a small example of how to get started here: https://realpython.com/flask-by-example-part-1-project-setup/.
2. Test your dummy API by calling the `ping` endpoint using your browser (or the `requests` Python library).
3. Implement the prediction endpoint, which takes a JSON dataset and returns predictions also in JSON. Note that the prediction endpoint needs to be able to load a trained model (from a pkl file) to be able to do predictions.

Once you're done with the prediction endpoint, you should be able to send predictions to the API using something like this:

```
import pandas as pd
import requests

predict_df = pd.read_csv("../use_case/predict.csv")

response = requests.post("http://localhost:5000/predict", data=predict_df.to_json())

response.raise_for_status()

print(response.json())
```

*Bonus*

4. Add a *serve* CLI command to start your Flask application from the command line.

## Copyright

This material is copyright of GoDataDriven. Please don't spread or duplicate without obtaining explicit permission.
