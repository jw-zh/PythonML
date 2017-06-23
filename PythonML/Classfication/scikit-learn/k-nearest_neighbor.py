""" K-Nearest Neighbor 
KNN is a typical example of a lazy learner. 
It is called lazy not because of its apparent simplicity, 
but because it doesn't learn a discriminative function from the training data but memorizes the training dataset instead.

The KNN algorithm itself is fairly straightforward and can be summarized by the following steps:
-- Choose the number of k and a distance metric.
-- Find the k nearest neighbors of the sample that we want to classify.
-- Assign the class label by majority vote.
"""
import os
import sys
sys.path.append(os.getcwd())
from functions.plot_decision_regions import *
from test_data.iris import *

from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2, metric='minkowski')
knn.fit(data_train_std, catagory_train)
plot_decision_regions(data_combined_std, catagory_combined, classifier=knn, test_idx=range(105,150))
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.show()