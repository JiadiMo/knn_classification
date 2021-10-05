# KNN
import matplotlib.pyplot as plt
import time
import numpy as np
import operator
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.metrics import accuracy_score, f1_score
from preprocess import preprocess


def knn(trainData, testData, labels, k):
    """
    implement of knn
    :param trainData:  data used to measure input unclassified data
    :param testData: data need to be predicted
    :param labels: ground truth of data
    :param k: num of nerghbors
    """
    # Calculate row size of training set
    rowSize = trainData.shape[0]
    # Calculate difference between train sample and test sample
    diff = np.tile(testData, (rowSize, 1)) - trainData
    # Calculate the sum of squares of differences
    sqrDiff = diff ** 2
    sqrDiffSum = sqrDiff.sum(axis=1)
    # Calculate distances
    distances = sqrDiffSum ** 0.5
    # Sort distences
    sortDistance = distances.argsort()
    counts = {}

    for i in range(k):
        vote = labels[sortDistance[i]]
        counts[vote] = counts.get(vote, 0) + 1
    # Sorting the frequency of category occurrences from highest to lowest
    sortCount = sorted(counts.items(), key=operator.itemgetter(1), reverse=True)

    # Return the category with the highest number of occurrences
    return sortCount[0][0]


if __name__ == "__main__":
    X, y = preprocess()
    # Split dataset using StratifiedKFold
    skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
    print(skf)
    # Classification
    results = []
    f1 = []
    start = time.process_time()
    for train_index, test_index in skf.split(X, y):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        predicts = []
        for index in range(X_test.shape[0]):
            knnpre = knn(X_train, X_test[index, :], y_train, k=53)
            predicts.append(knnpre)
        acc = accuracy_score(y_test, np.array(predicts))
        f1_s = f1_score(y_test, np.array(predicts), average='weighted')
        results.append(acc)
        f1.append(f1_s)
    end = time.process_time()
    print("The processing time(s) is: ", end - start)
    print("Accuracy of KNN is: ", np.mean(results), ", f1-score is: ", np.mean(f1))
