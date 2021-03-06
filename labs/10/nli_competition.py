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
import sklearn.metrics
import sklearn.linear_model
import sklearn.neural_network

class Dataset:
    CLASSES = ["ARA", "DEU", "FRA", "HIN", "ITA", "JPN", "KOR", "SPA", "TEL", "TUR", "ZHO"]

    def __init__(self, name="nli_dataset.train.txt"):
        if not os.path.exists(name):
            raise RuntimeError("The {} was not found, please download it from ReCodEx".format(name))

        # Load the dataset and split it into `data` and `target`.
        self.data, self.prompts, self.levels, self.target = [], [], [], []
        with open(name, "r", encoding="utf-8") as dataset_file:
            for line in dataset_file:
                target, prompt, level, text = line.rstrip("\n").split("\t")
                self.data.append(text)
                self.prompts.append(prompt)
                self.levels.append(level)
                self.target.append(-1 if not target else self.CLASSES.index(target))
        self.target = np.array(self.target, np.int32)

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="nli_competition.model", type=str, help="Model path")

def main(args: argparse.Namespace):
    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()

        # TODO: Train a model on the given dataset and store it in `model`.
        model = None
        data = np.concatenate([np.array(train.data).reshape(-1,1), np.array(train.levels).reshape(-1, 1), np.array(train.prompts).reshape(-1,1)], axis=1)
        column_transformer = sklearn.compose.ColumnTransformer(
            (
                ['tfidf', sklearn.pipeline.FeatureUnion(
                    [
                    ("char", sklearn.feature_extraction.text.TfidfVectorizer(max_features=20000, analyzer='char', ngram_range=(3,6))),
                , 0],
                ['levels', sklearn.preprocessing.OneHotEncoder(), [1]],
                ['prompts', sklearn.preprocessing.OneHotEncoder(), [2]]
            )
        )
        pipeline = sklearn.pipeline.Pipeline([
            ('column_transformer', column_transformer), ('svm', sklearn.svm.LinearSVC(verbose=True))
            ])

        print(pipeline.get_params().keys())

        model = pipeline.fit(data, train.target)
        print(sklearn.metrics.accuracy_score(model.predict(data), train.target))

        


        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)
        # as a Python list or a NumPy array.
        predictions = model.predict(data)
        # print(sklearn.metrics.accuracy_score(predictions, test.target))

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)