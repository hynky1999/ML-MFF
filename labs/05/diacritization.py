#!/usr/bin/env python3
import argparse
import lzma
import pickle
import os
import urllib.request
from numpy.core.defchararray import isalpha
import sklearn.compose
import sklearn.datasets
import sklearn.model_selection
import sklearn.pipeline
import sklearn.preprocessing
import sklearn.linear_model
import sklearn.metrics

import numpy as np

class Dataset:
    LETTERS_NODIA = "acdeeinorstuuyz"
    LETTERS_DIA = "áčďéěíňóřšťúůýž"

    # A translation table usable with `str.translate` to rewrite characters with dia to the ones without them.
    DIA_TO_NODIA = str.maketrans(LETTERS_DIA + LETTERS_DIA.upper(), LETTERS_NODIA + LETTERS_NODIA.upper())

    def __init__(self,
                 name="fiction-train.txt",
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name))
            urllib.request.urlretrieve(url + name, filename=name)
            urllib.request.urlretrieve(url + name.replace(".txt", ".LICENSE"), filename=name.replace(".txt", ".LICENSE"))

        # Load the dataset and split it into `data` and `target`.
        with open(name, "r", encoding="utf-8-sig") as dataset_file:
            self.target = dataset_file.read()
        self.data = self.target.translate(self.DIA_TO_NODIA)



def translate_neigh(neigh, k):

    data_left = np.array(list(neigh[0:k]))
    data_left = np.where(np.char.isalpha(data_left), data_left.view(dtype=np.int32), 0)


    data_right = np.array(list(neigh[k+1:]))
    data_right = np.where(np.char.isalpha(data_right), data_right.view(dtype=np.int32), 0)

    last_space_i = np.argmin(data_left[::-1])
    data_left = np.concatenate([np.zeros([k-last_space_i]), data_left[k-last_space_i:]])
    first_space_i = np.argmin(data_right)
    data_right = np.concatenate([ data_right[:first_space_i], np.zeros([k-first_space_i])])


    return np.concatenate([data_left, np.array([neigh[k]]).view(dtype=np.int32), data_right, [0]])

def prepare_data(unprepared_data, target_data, allowed_letters, k):
    data = np.empty([0, 2*k+2])
    target = np.empty([0])
    # We don't want to recalculate index
    unprepared_data = k*" " + unprepared_data + k*" "
    target_data = k*" " + target_data + k*" "
    word_length = 0
    for i in range(len(unprepared_data)):
        print(i)
        if not isalpha(unprepared_data[i]):
            if(word_length > 0):
                data[-word_length:, -1] = word_length
                word_length = 0

        else:
            word_length += 1



        if not unprepared_data[i] in allowed_letters:
            continue

        # Only alpha chars and zeros
        row = translate_neigh(unprepared_data[i-k:i+k+1], k)
        data = np.concatenate([data, row.reshape([1,-1])], axis=0)
        target = np.concatenate([target, np.array([target_data[i]]).view(dtype=np.int32)])

    return data, target
        

        
def predict_word(model, word_buffer, data, allowed_letters, k):
    word = []
    for index in word_buffer:
        if not data[index] in allowed_letters:
            word.append(data[index])
        else:
            translated_dato = translate_neigh(data[index-k: index+k+1],k)
            translated_dato[k*2 + 1] = len(word_buffer)
            prediction = chr(int(model.predict(translated_dato.reshape([1,-1]))))
            word.append(prediction)
    return word
        



def predict(model, data, allowed_letters, k):
    prediction = []
    word_buffer = []

    data = k*" " + data + k*" "
    for i in range(len(data)):
        if not isalpha(data[i]):
            if(len(word_buffer) > 0):
                prediction.extend(predict_word(model, word_buffer, data, allowed_letters, k))
                word_buffer = []

            prediction.append(data[i])
            continue

        word_buffer.append(i)
    return ''.join(prediction[k:-k])
    

    
    

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--predict", default=None, type=str, help="Run prediction on given data")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.
parser.add_argument("--model_path", default="diacritization.model", type=str, help="Model path")

def main(args: argparse.Namespace):
    neigh_size = 2

    if args.predict is None:
        # We are training a model.
        np.random.seed(args.seed)
        train = Dataset()
        transformed_data_file_name = "transformed_data.txt"
        transformed_target_file_name = "transformed_target.txt"


        data = None
        target = None
        data_selection = slice(0, len(train.data))
        test_selection = slice(60000,70000)
        if not os.path.exists(transformed_data_file_name):
            data, target = prepare_data(train.data[data_selection], train.target[data_selection],Dataset.LETTERS_DIA + Dataset.LETTERS_NODIA, neigh_size,)
            with lzma.open(transformed_data_file_name, "wb") as data_file:
                pickle.dump(data, data_file)

            with lzma.open(transformed_target_file_name, "wb") as target_file:
                pickle.dump(target, target_file)

        else:
            with lzma.open(transformed_data_file_name, "rb") as data_file:
                data = pickle.load(data_file)

            with lzma.open(transformed_target_file_name, "rb") as target_file:
                target = pickle.load(target_file)

        one_hot = sklearn.preprocessing.OneHotEncoder(handle_unknown="ignore")
        standar_scaler = sklearn.preprocessing.StandardScaler()
        column_enc = sklearn.compose.ColumnTransformer(
            [
                ("one_hot", one_hot, slice(0, neigh_size+1)), 
                ("standar_scaler", standar_scaler, slice(neigh_size+1, neigh_size+2))
            ]
        )
        pipe = sklearn.pipeline.Pipeline([
            ("column_enc", column_enc),
            ("polynomial", sklearn.preprocessing.PolynomialFeatures()),
            ("regression", sklearn.linear_model.LogisticRegression(random_state=args.seed, max_iter=150, multi_class="multinomial", verbose=True))
            ])
        
        grid = sklearn.model_selection.GridSearchCV(pipe, [{
            'polynomial__degree': [1, 2],
            'regression__C': [0.01, 1, 100]
        }], cv=3, verbose=3)

        model = grid.fit(data, target)
        predictions = predict(model, train.data[test_selection], Dataset.LETTERS_DIA + Dataset.LETTERS_NODIA, neigh_size)
        print(accuracy(train.target[test_selection] ,predictions))


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
        predictions = predict(model, test.data, Dataset.LETTERS_DIA + Dataset.LETTERS_NODIA, neigh_size)

        return predictions


if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
