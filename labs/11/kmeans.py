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
from numpy.core.fromnumeric import _squeeze_dispatcher

import sklearn.datasets

parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--clusters", default=5, type=int, help="Number of clusters")
parser.add_argument("--examples", default=150, type=int, help="Number of examples")
parser.add_argument("--init", default="random", type=str, help="Initialization (random/kmeans++)")
parser.add_argument("--iterations", default=5, type=int, help="Number of kmeans iterations to perfom")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=51, type=int, help="Random seed")
# If you add more arguments, ReCodEx will keep them with your default values.

def plot(args: argparse.Namespace, iteration: int,
         data: np.ndarray, centers: np.ndarray, clusters: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    if args.plot is not True:
        if not plt.gcf().get_axes(): plt.figure(figsize=(4*2, 5*6))
        plt.subplot(6, 2, 1 + len(plt.gcf().get_axes()))
    plt.title("KMeans Initialization" if not iteration else
              "KMeans After Iteration {}".format(iteration))
    plt.gca().set_aspect(1)
    plt.scatter(data[:, 0], data[:, 1], c=clusters)
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=200, c="#ff0000")
    plt.scatter(centers[:, 0], centers[:, 1], marker="P", s=50, c=range(args.clusters))
    if args.plot is True: plt.show()
    else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

def main(args: argparse.Namespace) -> np.ndarray:
    # Create a random generator with a given seed
    generator = np.random.RandomState(args.seed)

    # Generate artificial data
    data, target = sklearn.datasets.make_blobs(
        n_samples=args.examples, centers=args.clusters, n_features=2, random_state=args.seed)

    # TODO: Initialize `centers` to be
    # - if args.init == "random", K random data points, using the indices
    #   returned by
    #     generator.choice(len(data), size=args.clusters, replace=False)
    # - if args.init == "kmeans++", generate the first cluster by
    #     generator.randint(len(data))
    #   and then iteratively sample the rest of the clusters proportionally to
    #   the square of their distances to their closest cluster using
    #     generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances))
    #   Use the `np.linalg.norm` to measure the distances.

    centers_i = []

    if args.init == 'random':
        centers_i = generator.choice(len(data), size=args.clusters, replace=False)

    if args.init == 'kmeans++':
        centers_i.append(generator.randint(len(data)))
        unused_points_indices = [i for i in range(len(data))]
        unused_points_indices.remove(centers_i[-1])
        for _ in range(args.clusters - 1):
            square_distances = [ min([np.linalg.norm(data[i_used] - data[i_unused])**2 for i_used in centers_i]) for i_unused in unused_points_indices]
            centers_i.append(generator.choice(unused_points_indices, p=square_distances / np.sum(square_distances)))
            unused_points_indices.remove(centers_i[-1])

    centers = data[centers_i, :]
    if args.plot:
        plot(args, 0, data, centers, clusters=None)


    # Run `args.iterations` of the K-Means algorithm.
    clusters = []
    for iteration in range(args.iterations):
        # TODO: Perform a single iteration of the K-Means algorithm, storing
        # zero-based cluster assignment to `clusters`.
        clusters = np.array([np.argmin([np.linalg.norm(center - dato) for center in centers]) for dato in data])

        centers = np.array([np.sum(data[np.argwhere(clusters == i).flatten()], axis=0)/np.argwhere(clusters == i).shape[0] for i in range(len(centers))])




        if args.plot:
            plot(args, 1 + iteration, data, centers, clusters)

    return clusters.flatten().tolist()

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    centers = main(args)
    print("Cluster assignments:", centers, sep="\n")
