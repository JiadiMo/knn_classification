# Weighted KNN (gaussian)
import time
import numpy as np
import matplotlib.pyplot as plt
import operator
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score
from preprocess import preprocess


def gaussian(dist, sigma=10.0):
    # Using gaussian to calculate weights
    weight = np.exp(-dist ** 2 / (2 * sigma ** 2))
    return weight


def w_knn(train_data, test_data, labels, k):
    """
    implement of weighted_knn
    :param trainData:  data used to measure input unclassified data
    :param testData: data need to be predicted
    :param labels: ground truth of data
    :param k: num of nerghbors
    """
    # Calculate row size of training set
    row_size = train_data.shape[0]
    # Calculate difference between train sample and test sample
    diff = np.tile(test_data, (row_size, 1)) - train_data
    # Calculate the sum of squares of differences
    sqrt_diff = diff ** 2
    sqrt_diff_sum = sqrt_diff.sum(axis=1)
    # Calculate distances
    distances = sqrt_diff_sum ** 0.5
    # Sort distences
    sort_distance = distances.argsort()
    counts = {}
    for i in range(k):
        w = gaussian(distances[sort_distance[i]])
        vote = labels[sort_distance[i]]
        counts[vote] = counts.get(vote, 0) + w
    # Sorting the frequency of category occurrences from highest to lowest
    sort_counts = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

    # Return the category with the highest number of occurrences
    return sort_counts[0][0]


if __name__ == "__main__":
    X, y = preprocess()
    # Split dataset using StratifiedKFold
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    print(skf)
    # Classification
    results = []
    f1s = []
    start = time.process_time()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        predicts = []
        for index in range(X_test.shape[0]):
            predict = w_knn(X_train, X_test[index, :], y_train, k=80)
            predicts.append(predict)
        f1 = f1_score(y_test, np.array(predicts), average='weighted')
        acc = accuracy_score(y_test, np.array(predicts))
        results.append(acc)
        f1s.append(f1)

    end = time.process_time()
    print("The processing time(s) is: ", end - start)
    print("Accuracy of Weighted KNN is: ", np.mean(results), ", f1-score is: ", np.mean(f1s))

