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
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--C", default=1, type=float, help="Inverse regularization strength")
parser.add_argument("--data_size", default=200, type=int, help="Data size")
parser.add_argument("--kernel", default="poly", type=str, help="Kernel type [poly|rbf]")
parser.add_argument("--kernel_degree", default=1, type=int, help="Degree for poly kernel")
parser.add_argument("--kernel_gamma", default=1.0, type=float, help="Gamma for poly and rbf kernel")
parser.add_argument("--max_iterations", default=1000, type=int, help="Maximum number of iterations to perform")
parser.add_argument("--max_passes_without_as_changing", default=10, type=int, help="Number of passes without changes to stop after")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
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
    y = 0
    for i in range(vectors.shape[0]):
        y += a[i] * t[i] * kernel(args, x,vectors[i])
    return y + b



def smo(
    args: argparse.Namespace,
    train_data: np.ndarray, train_target: np.ndarray,
    test_data: np.ndarray, test_target: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Create initial weights
    a, b = np.zeros(len(train_data)), 0
    generator = np.random.RandomState(args.seed)

    passes_without_as_changing = 0
    train_accs, test_accs = [], []
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

            d_i = train_data[i]
            d_j = train_data[j]


            # TODO: Check that a[i] fulfils the KKT conditions, using `args.tolerance` during comparisons.
            E_i = predict(args, train_data[i], train_data, a, train_target,b) - t_i
            E_j = predict(args, train_data[j], train_data, a, train_target,b) - t_j
            if not((a_i < args.C - args.tolerance and E_i * t_i < -args.tolerance) or (a_i > args.tolerance and t_i * E_i > args.tolerance)):
                continue






            # If the conditions do not hold, then
            # - compute the updated unclipped a_j^new.
            #
            dL_da_j = t_j * (E_i - E_j)
            dL_da_j_2 = 2*kernel(args,d_i,d_j) - kernel(args,d_i, d_i) - kernel(args,d_j,d_j)


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

            b_i_new = b - E_i - t_i*(a_i_new - a_i)*kernel(args,d_i,d_i) - t_j*(a_j_new - a_j)*kernel(args,d_j,d_i)
            b_j_new = b - E_j - t_i*(a_i_new - a_i) * kernel(args,d_i,d_j) - t_j*(a_j_new - a_j)*kernel(args,d_j,d_j)




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

        train_pred = []
        test_pred = []
        for i in range(len(train_data)):
            if predict(args, train_data[i], train_data, a, train_target, b) > 0:
                train_pred.append(1)
            else:
                train_pred.append(-1)
        for i in range(len(test_data)):
            if predict(args, test_data[i], train_data, a, train_target, b) > 0:
                test_pred.append(1)
            else:
                test_pred.append(-1)
        train_accs.append(sklearn.metrics.accuracy_score(train_pred, train_target))
        test_accs.append(sklearn.metrics.accuracy_score(test_pred, test_target))



        # Stop training if max_passes_without_as_changing passes were reached
        passes_without_as_changing = 0 if as_changed else passes_without_as_changing + 1
        if passes_without_as_changing >= args.max_passes_without_as_changing:
            break

        if len(train_accs) % 100 == 0 and len(train_accs) < args.max_iterations:
            print("Iteration {}, train acc {:.1f}%, test acc {:.1f}%".format(
                len(train_accs), 100 * train_accs[-1], 100 * test_accs[-1]))

    # TODO: Create an array of support vectors (in the same order in which they appeared
    # in the training data; to avoid rounding errors, consider a training example
    # a support vector only if a_i > `args.tolerance`) and their weights (a_i * t_i).
    support_vectors, support_vector_weights = [], []
    for i in range(len(a)):
        if(a[i] > args.tolerance):
            support_vectors.append(train_data[i])
            support_vector_weights.append(train_target[i] * a[i])


    print("Done, iteration {}, support vectors {}, train acc {:.1f}%, test acc {:.1f}%".format(
        len(train_accs), len(support_vectors), 100 * train_accs[-1], 100 * test_accs[-1]))

    return support_vectors, support_vector_weights, b, train_accs, test_accs

def main(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, list[float], list[float]]:
    # Generate an artifical regression dataset, with +-1 as targets
    data, target = sklearn.datasets.make_classification(
        n_samples=args.data_size, n_features=2, n_informative=2, n_redundant=0, random_state=args.seed)
    target = 2 * target - 1

    # Split the dataset into a train set and a test set.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Run the SMO algorithm
    support_vectors, support_vector_weights, bias, train_accs, test_accs = smo(
        args, train_data, train_target, test_data, test_target)

    if args.plot:
        import matplotlib.pyplot as plt
        def plot(predict, support_vectors):
            xs = np.linspace(np.min(data[:, 0]), np.max(data[:, 0]), 50)
            ys = np.linspace(np.min(data[:, 1]), np.max(data[:, 1]), 50)
            predictions = [[predict(np.array([x, y])) for x in xs] for y in ys]
            test_mismatch = np.sign([predict(x) for x in test_data]) != test_target
            plt.figure()
            plt.contourf(xs, ys, predictions, levels=0, cmap=plt.cm.RdBu)
            plt.contour(xs, ys, predictions, levels=[-1, 0, 1], colors="k", zorder=1)
            plt.scatter(train_data[:, 0], train_data[:, 1], c=train_target, marker="o", label="Train", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(support_vectors[:, 0], support_vectors[:, 1], marker="o", s=90, label="Support Vectors", c="#00dd00")
            plt.scatter(test_data[:, 0], test_data[:, 1], c=test_target, marker="*", label="Test", cmap=plt.cm.RdBu, zorder=2)
            plt.scatter(test_data[test_mismatch, 0], test_data[test_mismatch, 1], marker="*", s=130, label="Test Errors", c="#ffff00")
            plt.legend(loc="upper center", ncol=4)

        # If you want plotting to work (not required for ReCodEx), you need to
        # define `predict_function` computing SVM value `y(x)` for the given x.
        predict_function = lambda x: None

        plot(predict_function, support_vectors)
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    return support_vectors, support_vector_weights, bias, train_accs, test_accs

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    main(args)
