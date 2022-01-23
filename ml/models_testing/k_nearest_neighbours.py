import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from sklearn.model_selection import train_test_split


def single_test(X, y, n):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

    #data -> 
    #   X_train - > [[R,G,B], ...] -> list of list of values | train set
    #   Y_train - > [Color, ...] -> list of values | train set
    #   X_test - > [[R,G,B], ...] -> list of list of values | test set
    #   Y_test - > [Color, ...] -> list of values | test set

    classifier = KNeighborsClassifier(n_neighbors = n)
    classifier.fit(X_train, y_train)

    y_predicted = classifier.predict(X_test)

    n_good_predictions = 0.0
    for p, t in zip(y_predicted, y_test):
        if p == t:
            n_good_predictions += 1.0
    
    result_accuracy = n_good_predictions/len(y_test)

    #print(f'N -> {n}   Accuracy -> {result_accuracy}')

    return result_accuracy


def perform_test(X, y, n_min, n_max, iters_per_n):

    accuracy_per_n_neighbors = {}

    for n in range(n_min, n_max + 1, 1):

        accuracy = 0.0

        for _ in range(0, iters_per_n, 1):

            accuracy += single_test(X, y, n)

        accuracy /= iters_per_n
        accuracy_per_n_neighbors[n] = accuracy

    return accuracy_per_n_neighbors