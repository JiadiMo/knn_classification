# Ball-Tree (Sklearn)
import time
import numpy as np
from preprocess import preprocess
from sklearn.model_selection import StratifiedKFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score

X, y = preprocess()
# Split dataset using StratifiedKFold
skf = StratifiedKFold(n_splits=10, random_state=None, shuffle=False)
print(skf)
# Classification
start = time.process_time()

ball_tree = KNeighborsClassifier(n_neighbors=83, weights='uniform', algorithm='ball_tree', leaf_size=30,
                                 p=2, metric='minkowski', metric_params=None, n_jobs=1)
results = []
f1s = []
for train_index, test_index in skf.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    ball_tree.fit(X_train, y_train)
    predicts = ball_tree.predict(X_test)
    f1 = f1_score(y_test, predicts, average='weighted')
    acc = accuracy_score(y_test, predicts)
    results.append(acc)
    f1s.append(f1)

end = time.process_time()
print("The processing time(s) is: ", end - start)
print("Accuracy of Weighted KNN is: ", np.mean(results), ", f1-score is: ", np.mean(f1s))
