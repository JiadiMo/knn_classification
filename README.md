# knn_classification
According to the census income dataset, classical KNN algorithm and its improved algorithms are fully implemented to 
classify adult income. Then with the help to Sklearn, KD-Tree and Ball-Tree are implemented compared with KNN.

## Dataset
Census income dataset from https://archive.ics.uci.edu/ml/datasets/Census+Income. This dataset has 15 attributes
including income attribute (>50K, <=50K).

##	Data pre-processing
- Import datasets
- Missing values
- Duplicates and outliers 	 
- Labeling and Scaling
- StratifiedKFold

## KNN algorithm
### Environmental Setting
- Python 3.7
- macOS Big Sur

###  Algorithms
#### KNN [implement with optimal k=53](knn.py)
Calculate all distances between test set and training set and fine k nearest neighbors to vote for category.
#### Weighted KNN [implement with optimal k=80](weighted_knn.py)
Improve the classical KNN algorithm by considering a weight w to the k nearest neighbors according to their distance from the sample to be classified. The weight is calculated using gaussian to void extreme values.
#### KNN Mean [implement with optimal k=42](knnm.py)
Generalize the pattern of each category and reduce error influenced by outliers, specifically by calculating the average distance of the k nearest samples of each category from the test sample. Then, the predicted outcome is the same as the category closest to the test sample.
#### KD-Tree [implement with optimal k=103 in sklearn](KD_tree.py)
Storage of instance points in k-dimensional space for fast retrieval.
#### Ball-Tree [implement with optimal k=83 in sklearn](Ball_tree.py)
Splitting data over a series of nested hyperspheres.
### Comparison
| Algorithm      | Similarity metric     | k-value     | Accuracy | F1-score| Time|
| ---------- | :-----------:  | :-----------: |:-----------: |:-----------: |:-----------: |
|KNN	|Euclidean distance	|53|	0.8484|	0.8284	|1341.5s|
|Weighted KNN	|Euclidean distance|	80|	0.8487	|0.8226	|1399.4s|
|KNN Mean	|Euclidean distance	|42	|0.8490	|0.8263|	2876.1s|
|KD-Tree (Sklearn)|Euclidean distance|	103	|0.8484|	0.8204|	245.5s|
|Ball-Tree (Sklearn)|	Euclidean distance	|83	|0.8484	|0.8221|	206.5s|


