""" Sequential Backward Selection (SBS)
This algorithm aims to reduce the dimensionality of the initial feature subspace 
with a minimum decay in performance of the classifier 
to improve upon computational efficiency.
Steps:
1 - Initialize the algorithm with k = d, where d is the dimensionality of the full feature space Xd.
2 - Determine the feature x` that maximizes the criterion x` = arg max J(Xk - x) where x belongs to Xk.
3 - Remove the feature x` from the feature set.
4 - Terminate if k equals the number of desired features, if not, go to step 2.
"""

from sklearn.base import clone
from itertools import combinations
import numpy as np
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score, test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=self.test_size, random_state=self.random_state)
        dim = X_train.shape[1]
        self.indices_ = tuple(range(dim))
        self.subsets_ = [self.indices_]
        score = self._calc_score(X_train, y_train, X_test, y_test, self.indices_)
        self.scores_ = [score]
        while dim > self.k_features:
            scores = []
            subsets = []
            for p in combinations(self.indices_, r=dim-1):
                score = self._calc_score(X_train, y_train, X_test, y_test, p)
                scores.append(score)
                subsets.append(p)
            best = np.argmax(scores)
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
            dim -= 1
            self.scores_.append(scores[best])
        self.k_score_ = self.scores_[-1]
        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        return score

# Test with KNN classifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import os
import sys
sys.path.append(os.getcwd())
from test_data.wine import *

knn = KNeighborsClassifier(n_neighbors=2)
sbs = SBS(knn, k_features=1)
sbs.fit(data_train_std, catagory_train)

k_feat = [len(k) for k in sbs.subsets_]
plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
plt.show()

k5 = list(sbs.subsets_[8])
print(df_wine.columns[1:][k5])

knn.fit(data_train_std, catagory_train)
print('Training Accuracy: ', knn.score(data_train_std, catagory_train))
print('Test accuracy:', knn.score(data_test_std, catagory_test))

knn.fit(data_train_std[:, k5], catagory_train)
print('Training Accuracy: ', knn.score(data_train_std[:, k5], catagory_train))
print('Test accuracy:', knn.score(data_test_std[:, k5], catagory_test))
