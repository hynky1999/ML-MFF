#!/usr/bin/env python3
# Authors:
#
# Hynek Kydlicek
# bb506e12-05bd-11eb-9574-ea7484399335
#
# Ondrej Krsicka
# 7360531e-00a2-11eb-9574-ea7484399335
import argparse
from itertools import tee
from collections import deque
from sys import maxsize
from bisect import bisect
from typing import List, NamedTuple


import numpy as np
from numpy.core.defchararray import count
from numpy.core.numeric import argwhere
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--criterion", default="gini", type=str, help="Criterion to use; either `gini` or `entropy`")
parser.add_argument("--dataset", default="digits", type=str, help="Dataset to use")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--max_leaves", default=None, type=int, help="Maximum number of leaf nodes")
parser.add_argument("--min_to_split", default=2, type=int, help="Minimum examples required to split")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
# If you add more arguments, ReCodEx will keep them with your default values.



def main(args: argparse.Namespace) -> tuple[float, float]:

    class DecisionNode():
        def __init__(self, instances, depth):
            self.instances = instances
            self.depth = depth
            self.prediction = self.get_prediction(instances)
            self.left = None
            self.right = None
            if(args.criterion == 'gini'):
                self.criterion = self.gini_criterion(instances)
            else:
                self.criterion = self.entropy_criterion(instances)


        def split(self,feature_i, feature_value):
            self.decision_feature = feature_i
            self.decision_value = feature_value

            sorted_instances = sorted(self.instances,key=lambda x: x.dato[feature_i])
            right_half_index = bisect(list(map(lambda x: x.dato[feature_i],sorted_instances)), feature_value)
            self.left = DecisionNode(sorted_instances[:right_half_index], self.depth + 1)
            self.right = DecisionNode(sorted_instances[right_half_index:], self.depth + 1)


        def get_prediction(self, arr):
            class_array = list(map(lambda x: x.c_class, arr))
            values, counts = np.unique(class_array, return_counts=True)
            return  values[np.argmax(counts)]

        def gini_criterion(self, arr):
            class_array = list(map(lambda x: x.c_class, arr))
            _, counts = np.unique(class_array, return_counts=True)
            counts = np.array(counts)/len(class_array)
            return np.sum(counts * (1- counts)) * len(class_array)

        def entropy_criterion(self, arr):
            class_array = list(map(lambda x: x.c_class, arr))
            _, counts = np.unique(class_array, return_counts=True)
            counts = np.array(counts)/len(class_array)
            return np.sum(counts * np.log(counts)) * -len(class_array)

        def can_split(self):
            if self.criterion == 0:
                return False

            if args.min_to_split and len(self.instances) < args.min_to_split:
                return False

            if args.max_depth and self.depth >= args.max_depth:
                return False

            return True


        def smallest_split_criterion(self):
            best_feature, best_value, best_crit_value = 0,0, maxsize
            for i in range(len(self.instances[0].dato)):
                sorted_instances = sorted(self.instances,key=lambda x: x.dato[i])

                left_side = []
                right_side = deque(sorted_instances);
                while len(right_side) > 1:
                    left_side.append(right_side.popleft())
                    a = left_side[-1].dato[i]
                    b = right_side[0].dato[i]
                    if(a != b):
                        if(args.criterion == 'gini'):
                            val = self.gini_criterion(left_side) + self.gini_criterion(right_side)
                        else:
                            val = self.entropy_criterion(left_side) + self.entropy_criterion(right_side)

                        if(val < best_crit_value):
                            best_feature = i
                            best_value = (a+b)/2
                            best_crit_value = val
            return best_feature, best_value, best_crit_value


        def predict(self,x):
            if self.left != None:
                if(x[self.decision_feature] <= self.decision_value):
                    return self.left.predict(x)
                else:
                    return self.right.predict(x)
            
            return self.prediction


    class DatoWithClass(NamedTuple):
        dato : List[int]
        c_class : int




    # Use the given dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    instances = []
    for i in range(train_target.shape[0]):
        instances.append(DatoWithClass(train_data[i], train_target[i]))

    root = DecisionNode(instances, 0)
    if(args.max_leaves == None):
        stack = []
        stack.append(root)
        while(len(stack) > 0):
            node = stack.pop()
            if(not node.can_split()):
                continue

            best_feature, best_value, crit_value = node.smallest_split_criterion()
            node.split(best_feature, best_value)
            stack.append(node.right)
            stack.append(node.left)

    else:
        stack = []
        stack.append(root)
        while(len(stack) < args.max_leaves):
            best_feature, best_value, best_crit_value = 0,0, maxsize
            best_leaf = None
            for leaf in stack:
                if(not leaf.can_split()):
                    continue

                feature, value, crit_value = leaf.smallest_split_criterion()
                if(crit_value - leaf.criterion < best_crit_value):
                    best_crit_value = crit_value - leaf.criterion
                    best_feature = feature
                    best_value = value
                    best_leaf = leaf




            best_leaf.split(best_feature, best_value)
            stack.append(best_leaf.left)
            stack.append(best_leaf.right)
            stack.remove(best_leaf)

    # TODO: Manually create a decision tree on the training data.
    #
    # - For each node, predict the most frequent class (and the one with
    #   smallest index if there are several such classes).
    #
    # - When splitting a node, consider the features in sequential order, then
    #   for each feature consider all possible split points ordered in ascending
    #   value, and perform the first encountered split decreasing the criterion
    #   the most. Each split point is an average of two nearest unique feature values
    #   of the instances corresponding to the given node (e.g., for four instances
    #   with values 1, 7, 3, 3 the split points are 2 and 5).
    #
    # - Allow splitting a node only if:
    #   - when `args.max_depth` is not None, its depth must be less than `args.max_depth`;
    #     depth of the root node is zero;
    #   - there are at least `args.min_to_split` corresponding instances;
    #   - the criterion value is not zero.
    #
    # - When `args.max_leaves` is None, use recursive (left descendants first, then
    #   right descendants) approach, splitting every node if the constraints are valid.
    #   Otherwise (when `args.max_leaves` is not None), always split a node where the
    #   constraints are valid and the overall criterion value (c_left + c_right - c_node)
    #   decreases the most. If there are several such nodes, choose the one
    #   which was created sooner (a left child is considered to be created
    #   before a right child).

    # TODO: Finally, measure the training and testing accuracy.
    train_results = np.zeros([train_data.shape[0]])
    for i,dato in enumerate(train_data):
        train_results[i] = root.predict(dato)
    train_accuracy = sklearn.metrics.accuracy_score(train_results, train_target)

    test_results = np.zeros([test_data.shape[0]])
    for i,dato in enumerate(test_data):
        test_results[i] = root.predict(dato)

    test_accuracy = sklearn.metrics.accuracy_score(test_results, test_target)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
