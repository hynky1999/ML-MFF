#!/usr/bin/env python3
#
# All team solutions **must** list **all** members of the team.
# The members must be listed using their ReCodEx IDs anywhere
# in a comment block in the source file (on a line beginning with `#`).
#
# You can find out ReCodEx ID in the URL bar after navigating
# to your User profile page. The ID has the following format:
# 01234567-89ab-cdef-0123-456789abcdef.


# Authors:
#
# Hynek Kydlicek
# bb506e12-05bd-11eb-9574-ea7484399335
#
#
# Ondrej Krsicka
# 7360531e-00a2-11eb-9574-ea7484399335
import argparse
import enum
import lzma
import pickle
import os
import urllib.request
from numpy.core.fromnumeric import argmax
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.neural_network
import sklearn.metrics
from diacritization_eval import accuracy

import numpy as np
import re






def diac_to_numbers(l):
    if l in "áéíóúý":
        return 1
    if l in "čďěňřšťůž":
        return 2
    return 0

ZERO_CHAR = '\0'
translate_carka = str.maketrans("aeiouy", "áéíóúý")
translate_hacek = str.maketrans("cdenrstuz", "čďěňřšťůž")

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(
        LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(
                url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)

def prepare_data(unprepared_data, unprepared_target, allowed_letters, c, n):
    unprepared_data = str.translate(unprepared_data, str.maketrans('\n\0\t', "   ")).lower()

    data_text_vector = np.array(list(unprepared_data)).reshape([-1, 1])
    data_matrix = data_text_vector

    left_matrix = np.empty([data_matrix.shape[0],0])
    right_matrix = np.empty([data_matrix.shape[0],0])
    for i in range(8 + c):
        left_side = np.pad(np.delete(data_matrix[:, 0], list(range(-1, -i -2, -1))), (i+1, 0), constant_values=(
            ' ', ' '), mode='constant',).reshape([-1, 1])
        left_matrix = np.concatenate([left_matrix, left_side], axis=1)

        right_side = np.pad(np.delete(
            data_matrix[:, 0], list(range(0, i+1))), (0, i+1), constant_values=(' ', ' '), mode='constant',).reshape([-1, 1])
        right_matrix = np.concatenate([right_matrix, right_side], axis=1)




    indicies = []
    for i, letter in enumerate(data_matrix[:,0]):
        if letter in allowed_letters:
            indicies.append(i)
    
    left_inner_chars = np.full([len(indicies), c], ZERO_CHAR)
    left_outer_chars = np.full([len(indicies), n], ZERO_CHAR)
    for i,row in enumerate(left_matrix[indicies]):
        z = 0
        for z in range(0, len(row)):
            if row[z].isalpha():
                if z < c:
                    left_inner_chars[i][z] = row[z]

            else:
                break
        for l in range(z, min(len(row)-z , n)):
            left_outer_chars[i][l] = row[z+l]

    right_inner_chars = np.full([len(indicies), c], ZERO_CHAR)
    right_outer_chars = np.full([len(indicies), n], ZERO_CHAR)
    for i,row in enumerate(right_matrix[indicies]):
        z = 0
        for z in range(0, len(row)):
            if row[z].isalpha():
                if z < c:
                    right_inner_chars[i][z] = row[z]

            else:
                break
        for l in range(z, min(len(row)-z , n)):
            right_outer_chars[i][l] = row[z+l]

    data_matrix = np.concatenate([left_outer_chars, left_inner_chars, data_matrix[indicies, :], right_inner_chars, right_outer_chars], axis=1)
    data_matrix = data_matrix.view(np.int32)



    if(len(unprepared_target) != len(unprepared_data)):
        return data_matrix

    unprepared_target = unprepared_target.lower()
    target_text_vector = np.array(list(unprepared_target)).reshape([-1, 1])
    target_vector = np.zeros([len(indicies)])
    for i in range(len(indicies)):
        target_vector[i] = diac_to_numbers(target_text_vector[indicies[i], 0])

    return data_matrix, target_vector


def predict(model, data, allowed_letters,c, n):
    diac_predictions_proba = model.predict_log_proba(prepare_data(data, [], allowed_letters, c, n))


    diac_index = 0
    last_index = 0
    words = []
    for match in re.finditer("\w+", data):
        words.append(data[last_index: match.span()[0]])
        prediction_word = []
        for letter in match.group():
            pred_letter = letter
            if letter.lower() in allowed_letters:
                prediction = np.argmax(diac_predictions_proba[diac_index])
                if prediction == 1:
                    pred_letter = str.translate(letter, translate_carka)
                if prediction == 2:
                    pred_letter = str.translate(letter, translate_hacek)

                if letter.isupper():
                    pred_letter = pred_letter.upper()
                diac_index += 1

            prediction_word.append(pred_letter)





        words.append(''.join(prediction_word))
        last_index = match.span()[1]

    words.append(data[last_index:])
    return ''.join(words)




parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str,
                    help="Run prediction on given data")
parser.add_argument("--recodex", default=False,
                    action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument(
    "--model_path", default="diacritization.model", type=str, help="Model path")


def main(args: argparse.Namespace):
    core_size = 4
    neigh_size = 4

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        data_selection = slice(0, len(train.data))
        data, target = prepare_data(
            train.data[data_selection], train.target[data_selection], Dataset.LETTERS_NODIA, core_size, neigh_size)

        one_hot = sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")
        pipe = sklearn.pipeline.Pipeline([
            ("one_hot", one_hot),
            ("MLP", sklearn.neural_network.MLPClassifier(max_iter=30, verbose=True, hidden_layer_sizes=[500]))
        ])

        # grid = sklearn.model_selection.GridSearchCV(pipe, [{
        #     'MLP__hidden_layer_sizes' : [[700], [1000], [1500]],
        # }],
        # cv=3,
        # verbose=1)
        print("model fitting")

        model = pipe.fit(data, target)



        # mlp = model.best_estimator_["MLP"]
        # USE ABOVE WITH GRID
        mlp = model["MLP"]
        mlp._optimizer = None
        for i in range(len(mlp.coefs_)): mlp.coefs_[i] = mlp.coefs_[i].astype(np.float16)
        for i in range(len(mlp.intercepts_)): mlp.intercepts_[i] = mlp.intercepts_[i].astype(np.float16)
        
        # TODO: Train a model on the given dataset and store it in `model`.
        # Serialize the model.
        with lzma.open(args.model_path, "wb") as model_file:
            pickle.dump(model, model_file)

    else:
        # Use the model and return test set predictions.
        test = Dataset(args.predict)

        with lzma.open(args.model_path, "rb") as model_file:
            model = pickle.load(model_file)

        # TODO: Generate `predictions` with the test set predictions. Specifically,
        # produce a diacritized `str` with exactly the same number of words as `test.data`.
        predictions = predict(
            model, test.data, Dataset.LETTERS_NODIA,core_size, neigh_size)
        print(accuracy(predictions, test.target))


        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
