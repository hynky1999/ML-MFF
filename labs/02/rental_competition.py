#!/usr/bin/env python3
import argparse
import lzma
import os
import pickle
import urllib.request
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics

import numpy as np

class Dataset:
    """Rental Dataset.

    The dataset instances consist of the following 12 features:
    - season (1: winter, 2: spring, 3: summer, 4: autumn)
    - year (0: 2011, 1: 2012)
    - month (1-12)
    - hour (0-23)
    - holiday (binary indicator)
    - day of week (0: Sun, 1: Mon, ..., 6: Sat)
    - working day (binary indicator; a day is neither weekend nor holiday)
    - weather (1: clear, 2: mist, 3: light rain, 4: heavy rain)
    - temperature (normalized so that -8 Celsius is 0 and 39 Celsius is 1)
    - feeling temperature (normalized so that -16 Celsius is 0 and 50 Celsius is 1)
    - relative humidity (0-1 range)
    - windspeed (normalized to 0-1 range)

    The target variable is the number of rentals in the given hour.
    """
    def __init__(self,
                 name="rental_competition.train.npz",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)

        # Load the dataset and return the data and targets.
        dataset = np.load(name)
        for key, value in dataset.items():
            setattr(self, key, value)


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="rental_competition.model", type=str, help="Model path")

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        data = np.concatenate((train.data, np.ones([train.data.shape[0], 1])), axis=1)
        X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(data,train.target, test_size=0., random_state=args.seed)
        one_hot = sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")
        standar_scaler = sklearn.preprocessing.StandardScaler()
        column_enc = sklearn.compose.ColumnTransformer(
            [
                ("one_hot", one_hot, slice(0, 8)), 
                ("standar_scaler", standar_scaler, slice(8, X_test.shape[1]))
            ]
        )
        lambdas = np.geomspace(0.01, 10, num=500)
        tranformer = column_enc.fit(X_train)
        best_lambda = None
        best_rmse = None
        rmses = []
        for l in lambdas:
            model = sklearn.linear_model.SGDRegressor(alpha=l).fit(tranformer.transform(X_train), y_train)
            prediction = model.predict(tranformer.transform(X_test))
            rmse = sklearn.metrics.mean_squared_error(prediction, y_test, squared=False)
            if(best_rmse == None or best_rmse > rmse):
                best_rmse = rmse
                best_lambda = l
            rmses.append(l)

        print(best_rmse)
        enc = sklearn.pipeline.Pipeline([("column", column_enc), ("regression", sklearn.linear_model.SGDRegressor(alpha=best_lambda))] )
        model = enc.fit(X_train, y_train)

        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions, either as a Python list or a NumPy array.
        test = Dataset(args.predict)
        data = np.concatenate((test.data, np.ones([test.data.shape[0], 1])), axis=1)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions.
        predictions = model.predict(data)


        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
