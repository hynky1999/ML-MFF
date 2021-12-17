#!/usr/bin/env python3
# Authors:
#
# Hynek Kydlicek
# bb506e12-05bd-11eb-9574-ea7484399335
#
# Ondrej Krsicka
# 7360531e-00a2-11eb-9574-ea7484399335
import argparse
import dataclasses
from collections import namedtuple

import numpy as np

class ArtificialData:
    @dataclasses.dataclass
    class Sentence:
        """ Information about a single dataset sentence."""
        gold_edits: int # Number of required edits to be performed.
        predicted_edits: int # Number of edits predicted by a model.
        predicted_correct: int # Number of correct edits predicted by a model.
        human_rating: int # Human rating of the model prediction.

    def __init__(self, args: argparse.Namespace):
        generator = np.random.RandomState(args.seed)

        self.sentences = []
        for _ in range(args.data_size):
            gold = generator.poisson(2)
            correct = generator.randint(gold + 1)
            predicted = correct + generator.poisson(0.5)
            human_rating = max(0, int(100 - generator.uniform(5, 8) * (gold - correct) - generator.uniform(8, 13) * (predicted - correct)))
            self.sentences.append(self.Sentence(gold, predicted, correct, human_rating))


parser = argparse.ArgumentParser()
# These arguments will be set appropriately by ReCodEx, even if you change them.
parser.add_argument("--bootstrap_samples", default=100, type=int, help="Bootstrap samples")
parser.add_argument("--data_size", default=1000, type=int, help="Data set size")
parser.add_argument("--plot", default=False, const=True, nargs="?", type=str, help="Plot the predictions")
parser.add_argument("--recodex", default=False, action="store_true", help="Running in ReCodEx")
parser.add_argument("--seed", default=42, type=int, help="Random seed")
# For these and any other arguments you add, ReCodEx will keep your default value.

def main(args: argparse.Namespace) -> tuple[float, float]:
    # Create the artificial data
    data = ArtificialData(args)

    # Create `args.bootstrap_samples` bootstrapped samples of the dataset by
    # sampling sentences of the original dataset, and for each compute
    # - average of human ratings
    # - TP, FP, FN counts of the predicted edits
    human_ratings, predictions = [], []
    generator = np.random.RandomState(args.seed)
    for _ in range(args.bootstrap_samples):
        # Bootstrap sample of the dataset
        sentences = generator.choice(data.sentences, size=len(data.sentences), replace=True)

<<<<<<< HEAD
        # TODO: Append the averate of human ratings of `sentences` to `humans`.
        human_ratings.append(np.mean(list(map(lambda x: x.human_rating,sentences))))
=======
        # TODO: Append the average of human ratings of `sentences` to `humans`.
        human_ratings.append(...)
>>>>>>> e6505da59fbb036fec969502e355c21f05a7e858

        # TODO: Compute TP, FP, FN counts of predicted edits in `sentences`
        # and append them to `predictions`.

        TP = np.sum(list(map(lambda x: x.predicted_correct, sentences)))
        FP = np.sum(list(map(lambda x: x.predicted_edits - x.predicted_correct, sentences)))
        FN = np.sum(list(map(lambda x: x.gold_edits - x.predicted_correct, sentences)))
        score_tuple = namedtuple('score_tuple', 'TP FP FN')
        predictions.append(score_tuple(TP, FP, FN))

    # Compute Pearson correlation between F_beta score and human ratings
    # for betas between 0 and 2.
    betas, correlations = [], []
    for beta in np.linspace(0, 2, 201):
        betas.append(beta)
        f_betas = []
        for sample in predictions:
            F_beta = (1 + beta**2)*sample.TP/((1+beta**2)*sample.TP + beta**2 * sample.FN + sample.FP)
            f_betas.append(F_beta)

        # TODO: For each bootstap dataset, compute the F_beta score using
        # the counts in `predictions` and then manually compute the Pearson
        # correlation between the computed scores and `human_ratings`. Append
        # the result to `correlations`.
        upper = np.sum((np.array(f_betas) - np.mean(f_betas)) * (np.array(human_ratings) - np.mean(human_ratings)))
        lower = np.sqrt(np.sum(np.power(np.array(f_betas) - np.mean(f_betas), 2))) * np.sqrt(np.sum(np.power(np.array(human_ratings) - np.mean(human_ratings), 2)))
        correlations.append(upper/lower)

    if args.plot:
        import matplotlib.pyplot as plt
        plt.plot(betas, correlations)
        plt.xlabel(r"$\beta$")
        plt.ylabel(r"Pearson correlation of $F_\beta$-score and human ratings")
        if args.plot is True: plt.show()
        else: plt.savefig(args.plot, transparent=True, bbox_inches="tight")

    # TODO: Assign the highest correlation to `best_correlation` and
    # store corresponding beta to `best_beta`.
    best_arg = np.argmax(correlations)
    best_beta, best_correlation = betas[best_arg], correlations[best_arg]

    return best_beta, best_correlation

if __name__ == "__main__":
    args = parser.parse_args([] if "__file__" not in globals() else None)
    best_beta, best_correlation = main(args)

    print("Best correlation of {:.3f} was found for beta {:.2f}".format(
        best_correlation, best_beta))
