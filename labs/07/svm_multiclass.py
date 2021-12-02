#!/usr/bin/env python3
import argparse
from matplotlib.pyplot import cla

import numpy as np
from numpy.core.numeric import indices
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--classes", default=10, type=int, help="Number of classes")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.5, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--tolerance", default=1e-7, type=float, help="Default tolerance for KKT conditions")
# If you add more arguments, ReCodEx will keep them with your default values.

def kernel(args: argparse.Namespace, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    # TODO: As in `kernel_linear_regression`, We consider the following `args.kernel`s:
    # - "poly": K(x, y; degree, gamma) = (gamma * x^T y + 1) ^ degree
    # - "rbf": K(x, y; gamma) = exp^{- gamma * ||x - y||^2}
    if args.kernel == 'poly':
        return (args.kernel_gamma * np.dot(x,y) + 1)**args.kernel_degree
    elif args.kernel == 'rbf':
        return np.exp(-args.kernel_gamma*(np.dot(np.abs(x-y), np.abs(x-y))))


# We implement the SMO algorithm as a separate method, so we can use
# it in the svm_multiclass assignment too.
def predict(args, x, vectors, a, t, b):
    return np.sum(a * t * np.array( [kernel(args,x, vec) for vec in vectors])) + b

def predict_weights(args, x, vectors, weights, b):
    return np.sum(weights * np.array( [kernel(args,x, vec) for vec in vectors])) + b


def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)
    train_target = train_target.reshape([-1])

    passes_without_as_changing = 0
    train_accs, test_accs = [], []

    kernel_arr = np.zeros([train_data.shape[0], train_data.shape[0]])
    for i in range(train_data.shape[0]):
        for j in range(train_data.shape[0]):
            kernel_arr[i,j] = kernel(args, train_data[i], train_data[j])



    for _ in range(args.max_iterations):
        as_changed = 0
        # Iterate through the data
        for i, j in enumerate(generator.randint(len(a) - 1, size=len(a))):
            # We want j != i, so we "skip" over the value of i
            j = j + (j >= i)
            a_i = a[i]
            a_j = a[j]

            t_i = train_target[i]
            t_j = train_target[j]


            # TODO: Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.
            E_i = predict(args, train_data[i], train_data, a, train_target,b) - t_i
            E_j = predict(args, train_data[j], train_data, a, train_target,b) - t_j
            if not((a_i < args.C - args.tolerance and E_i * t_i < -args.tolerance) or (a_i > args.tolerance and t_i * E_i > args.tolerance)):
                continue






            # If the conditions do not hold, then
            # - compute the updated unclipped a_j^new.
            #
            dL_da_j = t_j * (E_i - E_j)
            dL_da_j_2 = 2*kernel_arr[i,j] - kernel_arr[i,i] - kernel_arr[j,j]


            #   If the second derivative of the loss with respect to a[j]
            #   is > -`args.tolerance`, do not update a[j] and continue
            #   with next i.

            if dL_da_j_2 > -args.tolerance:
                continue

            a_j_new = a_j - dL_da_j/dL_da_j_2
            # - clip the a_j^new to suitable [L, H].
            L = 0
            H = 0
            if(t_i == t_j):
                L = max(0, a_i + a_j - args.C)
                H = min(args.C, a_i + a_j)
            else:
                L = max(0, a_j - a_i)
                H = min(args.C, args.C + a_j - a_i)
            

            if a_j_new < L:
                a_j_new = L

            if a_j_new > H:
                a_j_new = H


             
            #
            #   If the clipped updated a_j^new differs from the original a[j]

            #   by less than `args.tolerance`, do not update a[j] and continue
            #   with next i.
            if abs(a_j_new - a_j) < args.tolerance:
                continue

            # - update a[j] to a_j^new, and compute the updated a[i] and b.
            a[j] = a_j_new
            a_i_new = a[i] - t_i*t_j*(a_j_new - a_j)
            a[i] = a_i_new

            b_i_new = b - E_i - t_i*(a_i_new - a_i)*kernel_arr[i,i] - t_j*(a_j_new - a_j)*kernel_arr[j,i]
            b_j_new = b - E_j - t_i*(a_i_new - a_i) * kernel_arr[i,j] - t_j*(a_j_new - a_j)*kernel_arr[j,j]




            #
            #   During the update of b, compare the a[i] and a[j] to zero by
            #   `> args.tolerance` and to C using `< args.C - args.tolerance`.
            if args.tolerance < a_i_new and a_i_new < args.C - args.tolerance:
                b = b_i_new
            elif args.tolerance < a_j_new and a_j_new < args.C - args.tolerance:
                b = b_j_new
            else:
                b = (b_i_new + b_j_new)/2

            # - increase `as_changed`
            as_changed += 1

        # TODO: After each iteration, measure the accuracy for both the
        # train set and the test set and append it to `train_accs` and `test_accs`.

        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    support_vectors, support_vector_weights = [], []

    indices = np.argwhere(a > args.tolerance).reshape([-1])
    support_vectors = train_data[indices]
    support_vector_weights = train_target[indices] * a[indices]



    return support_vectors, support_vector_weights, b, train_accs, test_accs

def main(args: argparse.Namespace) -> float:
    # Use the digits dataset.
    data, target = sklearn.datasets.load_digits(n_class=args.classes, return_X_y=True)
    data = sklearn.preprocessing.MinMaxScaler().fit_transform(data)

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # TODO: Using One-vs-One scheme, train (K \binom 2) classifiers, one for every
    # pair of classes $i < j$, using the `smo` method.
    #
    # When training a classifier for classes $i < j$:
    # - keep only the training data of these classes, in the same order
    #   as in the input dataset;
    # - use targets 1 for the class $i$ and -1 for the class $j$.

    class_votes = np.zeros([test_data.shape[0], args.classes])
    for i in range(args.classes):
        for j in range(i+1,args.classes):
            print(f"Traing classes {i} {j}")
            tmp_train_data = np.empty([0,train_data.shape[1]])
            tmp_train_target = np.empty([0,1])
            for z in range(len(train_data)):
                if(train_target[z] == i):
                    tmp_train_data = np.append(tmp_train_data, train_data[z].reshape(1, -1), axis=0)
                    tmp_train_target = np.append(tmp_train_target, np.array([[1]]), axis=0)

                if(train_target[z] == j):
                    tmp_train_data = np.append(tmp_train_data, train_data[z].reshape(1, -1), axis=0)
                    tmp_train_target = np.append(tmp_train_target, [[-1]], axis=0)


            vectors, weights, b = (smo(args, tmp_train_data, tmp_train_target, [], [])[:3])
            vectors = np.array(vectors)
            weights = np.array(weights)
            for z in range(len(test_data)):
                if predict_weights(args, test_data[z], vectors, weights, b) > 0:
                    class_votes[z,i] += 1
                else:
                    class_votes[z,j] += 1

    
            

    # TODO: Classify the test set by majority voting of all the trained classifiers,
    # using the lowest class index in the case of ties.
    #
    # Note that during prediction, only the support vectors returned by the `smo`
    # should be used, not all training data.
    #
    # Finally, compute the test set prediction accuracy.
    test_accuracy = sklearn.metrics.accuracy_score(np.argmax(class_votes, axis=1), test_target)

    return test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    accuracy = main(args)
    print("Test set accuracy: {:.2f}%".format(100 * accuracy))
