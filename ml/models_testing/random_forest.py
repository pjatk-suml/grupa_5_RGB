from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split


def single_test(X, y):

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
    #data -> 
    #   X_train - > [[R,G,B], ...] -> list of list of values | train set
    #   Y_train - > [Color, ...] -> list of values | train set
    #   X_test - > [[R,G,B], ...] -> list of list of values | test set
    #   Y_test - > [Color, ...] -> list of values | test set

    classifier = RandomForestClassifier()
    classifier.fit(X_train, y_train)

    good_predictions = 0.0
    y_predicted = classifier.predict(X_test)

    for p, t in zip(y_predicted, y_test):
        
        if p == t:
            good_predictions += 1.0

    accuracy = good_predictions / len(y_test)

    return accuracy

def perform_test(X, y, iter):

    average_accuracy = 0.0
    accuracies = []

    for _ in range(0, iter, 1):
        test_result = single_test(X, y)
        average_accuracy += test_result
        accuracies.append(test_result)

    average_accuracy /= iter

    return average_accuracy, accuracies