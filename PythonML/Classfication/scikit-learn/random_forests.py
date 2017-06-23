""" Random Forests
Intuitively, a random forest can be considered as an ensemble of decision trees. 
The idea behind ensemble learning is to combine weak learners to build a more robust model, 
a strong learner, that has a better generalization error and is less susceptible to overfitting. 
"""
import os
import sys
sys.path.append(os.getcwd())
from functions.plot_decision_regions import *
from test_data.iris import *

# Train with random forests.
from sklearn.ensemble import RandomForestClassifier
""" Description of Parameters
Trained a random forest from 10 decision trees via the n_estimators parameter, 
used the entropy criterion as an impurity measure to split the nodes, 
and used the n_jobs parameter for demonstration purposes, which allows us to parallelize the model training using multiple cores of our computer (here, two).
"""
forest = RandomForestClassifier(n_estimators=10, criterion='entropy', random_state=0, n_jobs=2)
forest.fit(data_train, catagory_train)

# Plot the results.
plot_decision_regions(data_combined, catagory_combined, classifier=forest, test_idx=range(105,150))
plt.xlabel('petal length')
plt.ylabel('petal width')
plt.legend(loc='upper left')
plt.show()