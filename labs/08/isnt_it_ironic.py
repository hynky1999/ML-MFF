#!/usr/bin/env python3
# Authors:
#
# Hynek Kydlicek
# bb506e12-05bd-11eb-9574-ea7484399335
#
# Ondrej Krsicka
# 7360531e-00a2-11eb-9574-ea7484399335
import argparse
import lzma
import pickle
import os
import urllib.request
import sys
from numpy.core.defchararray import mod
import sklearn.feature_extraction
import sklearn.naive_bayes
import sklearn.model_selection
import sklearn.pipeline
import sklearn.metrics

import numpy as np

class Dataset:
    def __init__(self,
                 name="isnt_it_ironic.train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        self.data = []
        self.target = []

        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            for line in dataset_file:
                label, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.target.append(int(label))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="isnt_it_ironic.model", type=str, help="Model path")

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = None
        pipe = sklearn.pipeline.Pipeline([
            ('count', sklearn.feature_extraction.text.CountVectorizer()),
            ('tfid', sklearn.feature_extraction.text.TfidfTransformer()),
            ("bayes", sklearn.naive_bayes.MultinomialNB())
        ])

        grid = sklearn.model_selection.GridSearchCV(pipe, [{
            "bayes__alpha": [0.1, 0.5, 0.75, 0.5, 1.2, 1, 10],
            "count__min_df": [1,2],
            "count__strip_accents": ["unicode", None],
            "count__lowercase": [True, False],
            "count__stop_words": [None, "english"],
            "count__token_pattern": [r"(?u)\b\w\w+\b",  r"(?u)(?:\b\w\w+\b)|(?:[\U0001F600-\U0001F64F]|[\U0001F300-\U0001F5FF]|[\U0001F680-\U0001F6FF])"],

        }], scoring="f1", verbose=3, cv=20)

        model = grid.fit(train.data, train.target)


        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)
        test.data = np.array(["Dogs are the best kind of humans"])

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions, either
        # as a Python list or a NumPy array.
        predictions = model.predict(test.data)
        print(predictions)
        # print(model.best_params_)
        # print(sklearn.metrics.f1_score(test.target, predictions))


        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
