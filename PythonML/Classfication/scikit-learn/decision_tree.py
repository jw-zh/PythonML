## Prepare dataset.
#from sklearn import datasets
#import numpy as np
#iris = datasets.load_iris()
#data = iris.data[:, [2, 3]]
#catagory = iris.target

## Split the dataset into separate training and testing datasets.
#from sklearn.cross_validation import train_test_split
## Parameter 'test_size' means to split data into 30 percent test data and 70 percent training data.
#data_train, data_test, catagory_train, catagory_test = train_test_split(data, catagory, test_size=0.3, random_state=0)

import os
import sys
sys.path.append(os.getcwd())
from functions.plot_decision_regions import *
from test_data.iris import *

# Train with decision tree model.
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier(criterion='entropy', max_depth=3, random_state=0)
tree.fit(data_train, catagory_train)
data_combined = np.vstack((data_train, data_test))
catagory_combined = np.hstack((catagory_train, catagory_test))


plot_decision_regions(data_combined, catagory_combined, classifier=tree, test_idx=range(105,150))
plt.xlabel('petal length [cm]')
plt.ylabel('petal width [cm]') 
plt.legend(loc='upper left')
plt.show()