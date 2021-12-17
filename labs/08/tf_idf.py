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
import sys
import urllib.request
import re

import numpy as np
import sklearn.metrics
import sklearn.model_selection
import sklearn.neighbors

class NewsGroups:
    def __init__(self,
                 name="20newsgroups.train.pickle",
                 data_size=None,
                 url="https://ufal.mff.cuni.cz/~straka/courses/npfl129/2122/datasets/"):
        if not os.path.exists(name):
            print("Downloading dataset {}...".format(name), file=sys.stderr)
            urllib.request.urlretrieve(url + name, filename=name)

        with lzma.open(name, "rb") as dataset_file:
            dataset = pickle.load(dataset_file)

        self.DESCR = dataset.DESCR
        self.data = dataset.data[:data_size]
        self.target = dataset.target[:data_size]
        self.target_names = dataset.target_names

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--idf", default=False, action="store_true", help="Use IDF weights")
parser.add_argument("--k", default=1, type=int, help="K nearest neighbors to consider")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=37, type=int, help="Random seed")
parser.add_argument("--tf", default=False, action="store_true", help="Use TF weights")
parser.add_argument("--test_size", default=1000, type=int, help="Test set size")
parser.add_argument("--train_size", default=1000, type=int, help="Train set size")
# For these and any other arguments you add, ReCodEx will keep your default value.

def main(args: argparse.Namespace) -> float:
    # Load the 20newsgroups data.
    newsgroups = NewsGroups(data_size=args.train_size + args.test_size)

    # Create train-test split.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        newsgroups.data, newsgroups.target, test_size=args.test_size, random_state=args.seed)

    # TODO: Create a feature for every word that is present at least twice
    # in the training data. A word is every sequence of at least 2 word characters,
    # where a word character corresponds to a regular expression `\w`.
    def get_features(data, allowed_words=[], idf_values = []):
        doc_words = []
        all_words = []

        for doc in data:
            words_in_doc = re.findall("\w\w+", doc)
            doc_words.append(words_in_doc)
            all_words.extend(words_in_doc)


    
        if not allowed_words:
            words, counts = np.unique( all_words, return_counts=True)
            unique_words = filter(lambda x: x[1] >= 2, zip(words, counts))
            unique_words_dict = { w:i for i, w in enumerate(map(lambda x: x[0], unique_words)) }
            allowed_words = unique_words_dict

        if not len(idf_values) > 0 and args.idf:
            idf_counts = np.zeros([len(allowed_words)])
            for doc in doc_words:
                words = np.unique(doc)
                for w in words:
                    w_index = allowed_words.get(w, -1)
                    if w_index != -1:
                        idf_counts[w_index] += 1;

            idf_values = np.log(len(doc_words)/(idf_counts + 1))


        features = []
        if args.idf and args.tf:
            features = get_idf_features(allowed_words, doc_words, idf_values) * get_tf_features(allowed_words, doc_words)

        if args.idf and not args.tf:
            features = get_idf_features(allowed_words, doc_words, idf_values)

        if not args.idf and args.tf:
            features = get_tf_features(allowed_words, doc_words)

        if not args.idf and not args.tf:
            features = get_bin_features(allowed_words, doc_words)

        
        return  features/np.linalg.norm(features, axis=1, ord=2).reshape([len(data),1]), allowed_words, idf_values

        
    def get_tf_features(word_index, docs):
        features = np.zeros([len(docs), len(word_index)])
        for doc_i in range(len(docs)):
            uniq_words, counts = np.unique(docs[doc_i], return_counts=True)
            for  uniq_i, uniq_w, in enumerate(uniq_words):
                w_index = word_index.get(uniq_w, -1)
                if(w_index != -1):
                    features[doc_i, w_index] = counts[uniq_i]
        return features


    def get_idf_features(word_index, docs, idf_values):

        features = np.zeros([len(docs), len(word_index)])
        for doc_i in range(len(docs)):
            for word in docs[doc_i]:
                w_index = word_index.get(word, -1)
                if(w_index != -1):
                    features[doc_i, w_index] = idf_values[w_index]

        return features
    
    def get_bin_features(word_index, docs):
        features = np.zeros([len(docs), len(word_index)])
        for doc_i in range(len(docs)):
            for word in docs[doc_i]:
                w_index = word_index.get(word, -1)
                if(w_index != -1):
                    features[doc_i, w_index] = 1
        return features

    # TODO: Weight the selected features using
    # - term frequency (TF), if `args.tf` is set;
    # - inverse document frequency (IDF), if `args.idf` is set; use
    #   the variant which contains `+1` in the denominator;
    # - TF * IDF, if both `args.tf` and `args.idf` are set;
    # - binary indicators, if nether `args.tf` nor `args.idf` are set.
    # Note that IDFs are computed on the train set and then reused without
    # modification on the test set, while TF is computed for every document separately.
    #
    # Finally, for each document L2-normalize its features.

    # TODO: Perform classification of the test set using the k-NN algorithm
    # from sklearn (pass the `algorithm="brute"` option), with `args.k` nearest
    # neighbors determined using the cosine similarity, where
    #   cosine_similarity(x, y) = x^T y / (||x|| * ||y||).
    # Note that for L2-normalized data (which we have), the nearest neighbors
    # are equivalent to using the usual Euclidean distance (L2 distance).
    
    
    train_features, allowed_words, idf_values = get_features(train_data)
    test_features, _, _ = get_features(test_data, allowed_words=allowed_words, idf_values=idf_values)
    model = sklearn.neighbors.KNeighborsClassifier(n_neighbors=args.k, algorithm="brute", metric="euclidean").fit(train_features, train_target)
    result = model.predict(test_features)

    # TODO: Evaluate the performance using macro-averaged F1 score.
    f1_score = sklearn.metrics.f1_score(test_target, result,average="macro")
    return f1_score

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    f1_score = main(args)
    print("F-1 score for TF={}, IDF={}, k={}: {:.1f}%".format(args.tf, args.idf, args.k, 100 * f1_score))
