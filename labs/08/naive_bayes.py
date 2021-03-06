#!/usr/bin/env python3
# Authors:
#
# Hynek Kydlicek
# bb506e12-05bd-11eb-9574-ea7484399335
#
# Ondrej Krsicka
# 7360531e-00a2-11eb-9574-ea7484399335

import argparse

import numpy as np
import scipy.stats

import sklearn.datasets
import sklearn.model_selection
import sklearn.metrics

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--alpha", default=0.1, type=float, help="Smoothing parameter for Bernoulli and Multinomial NB")
parser.add_argument("--naive_bayes_type", default="gaussian", type=str, help="NB type to use")
parser.add_argument("--classes", default=3, type=int, help="Number of classes")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> float:
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Train a naive Bayes classifier on the train data.
    #
    # The `args.naive_bayes_type` can be one of:
    # - "gaussian": implement Gaussian NB training, by estimating mean and
    #   variance of the input features. For variance estimation use
    #     1/N * \sum_x (x - mean)^2
    #   and additionally increase all estimated variances by `args.alpha`.
    weights = np.zeros([args.classes, 2, len(train_data[0])])
    per_class_matrices = []
    for i in range(args.classes):
        per_class_matrices.append(train_data[np.argwhere(train_target==i).flatten()])
    log_class_probs = np.zeros([args.classes])
    for i in range(args.classes):
        log_class_probs[i] = np.log(per_class_matrices[i].shape[0]/len(train_data))





    if args.naive_bayes_type == 'gaussian':
        for i in range(args.classes):
            weights[i] = [np.mean(per_class_matrices[i], axis=0), np.var(per_class_matrices[i],axis=0) + args.alpha]

    elif args.naive_bayes_type == 'bernoulli':
        for i in range(args.classes):
            weights[i][0] = (np.sum(np.where(per_class_matrices[i] >= 8, 1, 0), axis=0) + args.alpha)/(per_class_matrices[i].shape[0] + 2*args.alpha)

    else:
        for i in range(args.classes):
            weights[i][0] = (np.sum(per_class_matrices[i], axis=0) + args.alpha)/(np.sum(per_class_matrices[i]) + args.alpha * train_data.shape[1])

        
        


    #
    #   During prediction, you can compute probability density function of a Gaussian
    #   distribution using `scipy.stats.norm`, which offers `pdf` and `logpdf`
    #   methods, among others.
    #
    # - "multinomial": Implement multinomial NB with smoothing factor `args.alpha`.
    #
    # - "bernoulli": Implement Bernoulli NB with smoothing factor `args.alpha`.
    #   Because Bernoulli NB works with binary data, binarize the features as
    #   [feature_value >= 8], i.e., consider a feature as one iff it is >= 8,
    #   during both estimation and prediction.
<<<<<<< HEAD
    results = np.zeros([len(test_data)])

    if args.naive_bayes_type == 'gaussian':
        for i in range(results.shape[0]):
            test_values = np.zeros([args.classes])
            for class_i in range(args.classes):
                test_values[class_i] = np.sum(scipy.stats.norm.logpdf(test_data[i],loc=weights[class_i][0], scale=np.sqrt(weights[class_i][1]))) + log_class_probs[class_i]
            results[i] = np.argmax(test_values)


    if args.naive_bayes_type == 'bernoulli':
        for i in range(results.shape[0]):
            test_values = np.zeros([args.classes])
            for class_i in range(args.classes):
                test_values[class_i] = np.sum(scipy.stats.bernoulli.logpmf(np.where(test_data[i] >= 8, 1, 0) ,p=weights[class_i][0])) + log_class_probs[class_i]
            results[i] = np.argmax(test_values)

    if args.naive_bayes_type == 'multinomial':
        for i in range(results.shape[0]):
            test_values = np.zeros([args.classes])
            for class_i in range(args.classes):
                test_values[class_i] = scipy.stats.multinomial.logpmf(test_data[i], n=np.sum(test_data[i]) ,p=weights[class_i][0]) + log_class_probs[class_i]
            results[i] = np.argmax(test_values)

=======
    #
    # In all cases, the class prior is the distribution of the train data classes.
>>>>>>> e6505da59fbb036fec969502e355c21f05a7e858

    # TODO: Predict the test data classes and compute test accuracy.

    test_accuracy = sklearn.metrics.accuracy_score(test_target, results)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    test_accuracy = main(args)

    print("Test accuracy {:.2f}%".format(100 * test_accuracy))

