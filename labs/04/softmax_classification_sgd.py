#!/usr/bin/env python3
import argparse

import numpy as np
from numpy.core.numeric import zeros_like
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--batch_size", default=10, type=int, help="Batch size")
parser.add_argument("--classes", default=10, type=int, help="Number of classes to use")
parser.add_argument("--epochs", default=10, type=int, help="Number of SGD training epochs")
parser.add_argument("--learning_rate", default=0.01, type=float, help="Learning rate")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=797, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> tuple[np.ndarray, list[tuple[float, float]]]:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Use the digits dataset
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)

    # Append a constant feature with value 1 to the end of every input data
    data = np.pad(data, ((0, 0), (0, 1)), constant_values=1)

    # Split the dataset into a train set and a test set.
    # Use `sklearn.model_selection.train_test_split` method call, passing
    # arguments `test_size=args.test_size, random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Generate initial model weights
    weights = generator.uniform(size=[train_data.shape[1], args.classes], low=-0.1, high=0.1)

    for epoch in range(args.epochs):
        permutation = generator.permutation(train_data.shape[0])

        # TODO: Process the data in the order of `permutation`. For every
        # `args.batch_size` of them, average their gradient, and update the weights.
        # You can assume that `args.batch_size` exactly divides `train_data.shape[0]`.
        for i in range(train_data.shape[0]//args.batch_size):
            gradient = np.zeros([train_data.shape[1], args.classes]);
            for row in permutation[i*args.batch_size: (i+1)*args.batch_size]:
                y_roof = train_data[row] @ weights
                y = np.exp(y_roof)/np.sum(np.exp(y_roof))
                result = np.zeros(args.classes)
                result[train_target[row]] = 1
                gradient += np.atleast_2d(train_data[row]).T@np.atleast_2d(y-result)

            gradient = gradient/args.batch_size
            weights = weights - args.learning_rate * gradient
        #
        # Note that you need to be careful when computing softmax, because the exponentiation
        # in softmax can easily overflow. To avoid it, you should use the fact that
        # softmax(z) = softmax(z + any_constant) and compute softmax(z) = softmax(z - maximum_of_z).
        # That way we only exponentiate values which are non-positive, and overflow does not occur.


        # TODO: After the SGD epoch, measure the average loss and accuracy for both the
        # train test and the test set. The loss is the average MLE loss (i.e., the
        # negative log likelihood, or crossentropy loss, or KL loss) per example.

        def compute_prediction(x, y, weights):
            train_results = x@weights
            train_exp = np.exp(train_results -  np.atleast_2d(np.max(train_results, axis=1)).T)
        
            train_prediction = np.argmax(train_exp/np.atleast_2d(np.sum(train_exp, axis=1)).T, axis=1)
            train_accuracy = np.sum((np.where(np.abs(train_prediction - y) == 0, 1, 0)))/x.shape[0]
            return train_accuracy

        def compute_loss(x,y,weights):
                train_results = x@weights
                train_exp = np.exp(train_results -  np.atleast_2d(np.max(train_results, axis=1)).T)
                train_prediction = train_exp/np.atleast_2d(np.sum(train_exp, axis=1)).T
                train_selection = train_prediction[np.arange(x.shape[0]), [y]]
                loss = np.sum(-np.log(train_selection))/x.shape[0]
                return loss

        train_accuracy, train_loss, test_accuracy, test_loss = compute_prediction(train_data, train_target, weights), compute_loss(train_data, train_target, weights), compute_prediction(test_data, test_target, weights),  compute_loss(test_data, test_target, weights)



        print("After epoch {}: train loss {:.4f} acc {:.1f}%, test loss {:.4f} acc {:.1f}%".format(
            epoch + 1, train_loss, 100 * train_accuracy, test_loss, 100 * test_accuracy))

    return weights, [(train_loss, train_accuracy), (test_loss, test_accuracy)]

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    weights, metrics = main(args)
    print("Learned weights:", *(" ".join([" "] + ["{:.2f}".format(w) for w in row[:10]] + ["..."]) for row in weights.T), sep="\n")
