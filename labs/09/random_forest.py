#!/usr/bin/env python3
# Authors:
#
# Hynek Kydlicek
# bb506e12-05bd-11eb-9574-ea7484399335
#
# Ondrej Krsicka
# 7360531e-00a2-11eb-9574-ea7484399335
import argparse
from bisect import bisect
from collections import deque
from sys import maxsize
from typing import List, NamedTuple

import numpy as np
from numpy.core.fromnumeric import argmax
from sklearn import tree
import sklearn.datasets
import sklearn.metrics
import sklearn.model_selection

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bagging", default=False, action="store_true", help="Perform bagging")
parser.add_argument("--dataset", default="wine", type=str, help="Dataset to use")
parser.add_argument("--feature_subsampling", default=1, type=float, help="What fraction of features to subsample")
parser.add_argument("--max_depth", default=None, type=int, help="Maximum decision tree depth")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
parser.add_argument("--test_size", default=0.25, type=lambda x:int(x) if x.isdigit() else float(x), help="Test set size")
parser.add_argument("--trees", default=1, type=int, help="Number of trees in the forest")
# If you add more arguments, ReCodEx will keep them with your default values.

def main(args: argparse.Namespace) -> tuple[float, float]:
    # Use the given dataset
    data, target = getattr(sklearn.datasets, "load_{}".format(args.dataset))(return_X_y=True)

    # Split the data randomly to train and test using `sklearn.model_selection.train_test_split`,
    # with `test_size=args.test_size` and `random_state=args.seed`.
    train_data, test_data, train_target, test_target = sklearn.model_selection.train_test_split(
        data, target, test_size=args.test_size, random_state=args.seed)

    # Create random generators
    generator_feature_subsampling = np.random.RandomState(args.seed)
    def subsample_features(number_of_features:int) -> np.ndarray:
        return generator_feature_subsampling.uniform(size=number_of_features) <= args.feature_subsampling

    generator_bootstrapping = np.random.RandomState(args.seed)
    def bootstrap_dataset(train_data:np.ndarray) -> np.ndarray:
        return generator_bootstrapping.choice(len(train_data), size=len(train_data), replace=True)

    # TODO: Create a random forest on the trainining data.

    class DecisionNode():
        def __init__(self, instances, depth):
            self.instances = instances
            self.depth = depth
            self.prediction = self.get_prediction(instances)
            self.left = None
            self.right = None
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

            if len(self.instances) < 2:
                return False

            if args.max_depth and self.depth >= args.max_depth:
                return False

            return True


        def smallest_split_criterion(self):
            best_feature, best_value, best_crit_value = 0,0, maxsize
            feature_map = subsample_features(len(self.instances[0].dato))
            for i in range(len(self.instances[0].dato)):
                if(feature_map[i] == False):
                    continue

                sorted_instances = sorted(self.instances,key=lambda x: x.dato[i])

                left_side = []
                right_side = deque(sorted_instances);
                while len(right_side) > 1:
                    left_side.append(right_side.popleft())
                    a = left_side[-1].dato[i]
                    b = right_side[0].dato[i]
                    if(a != b):
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

    class DecisionTreeMaker():
        def __init__(self, data, target):

            instances = []
            for i in range(data.shape[0]):
                instances.append(DatoWithClass(data[i], target[i]))

            self.root = DecisionNode(instances, 0)


        def build(self):
            stack = []
            stack.append(self.root)
            while(len(stack) > 0):
                node = stack.pop()
                if(not node.can_split()):
                    continue

                best_feature, best_value, crit_value = node.smallest_split_criterion()
                node.split(best_feature, best_value)
                stack.append(node.right)
                stack.append(node.left)

        def predict(self,x):
            return self.root.predict(x)



    class DatoWithClass(NamedTuple):
        dato : List[int]
        c_class : int




    #
    # Use a simplified decision tree from the `decision_tree` assignment:
    # - use `entropy` as the criterion
    # - use `max_depth` constraint, so split a node only if:
    #   - its depth is less than `args.max_depth`
    #   - the criterion is not 0 (the corresponding instance targetsare not the same)
    # When splitting nodes, proceed in the depth-first order, splitting all nodes
    # in left subtrees before nodes in right subtrees.
    #
    # Additionally, implement:
    # - feature subsampling: when searching for the best split, try only
    #   a subset of features. Notably, when splitting a node (i.e., when the
    #   splitting conditions [depth, criterion != 0] are satisfied), start by
    #   generating a feature mask using
    #     subsample_features(number_of_features)
    #   which gives a boolean value for every feature, with `True` meaning the
    #   feature is used during best split search, and `False` it is not
    #   (i.e., when feature_subsampling == 1, all features are used).
    #
    # - train a random forest consisting of `args.trees` decision trees


    #
    # - if `args.bagging` is set, before training each decision tree
    #   create a bootstrap sample of the training data by calling
    #     dataset_indices = bootstrap_dataset(train_data)
    #   and if `args.bagging` is not set, use the original training data.
    #
    # During prediction, use voting to find the most frequent class for a given
    # input, choosing the one with smallest class index in case of a tie.

    # TODO: Finally, measure the training and testing accuracy.
    trees = []
    for _ in range(args.trees):
        tree_data = train_data
        tree_target = train_target
        if(args.bagging):
            indexes = bootstrap_dataset(train_data)
            tree_data = train_data[indexes, :]
            tree_target = train_target[indexes]
        tree = DecisionTreeMaker(tree_data, tree_target)
        tree.build()
        trees.append(tree)
    
    train_results = []
    for dato in train_data:
        train_results.append(np.argmax(np.bincount([tr.predict(dato) for tr in trees])))

    test_results = []
    for dato in test_data:
        test_results.append(np.argmax(np.bincount([tr.predict(dato) for tr in trees])))


    train_accuracy = sklearn.metrics.accuracy_score(train_results, train_target)
    test_accuracy = sklearn.metrics.accuracy_score(test_results, test_target)

    return train_accuracy, test_accuracy

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    train_accuracy, test_accuracy = main(args)

    print("Train accuracy: {:.1f}%".format(100 * train_accuracy))
    print("Test accuracy: {:.1f}%".format(100 * test_accuracy))
