# Prepare dataset.
from sklearn import datasets
import numpy as np
iris = datasets.load_iris()
data = iris.data[:, [2, 3]]
catagory = iris.target

# Split the dataset into separate training and testing datasets.
from sklearn.cross_validation import train_test_split
# Parameter 'test_size' means to split data into 30 percent test data and 70 percent training data.
data_train, data_test, catagory_train, catagory_test = train_test_split(data, catagory, test_size=0.3, random_state=0)

# Standardize the dataset.
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
sc.fit(data_train)
data_train_std = sc.transform(data_train)
data_test_std = sc.transform(data_test)

data_combined = np.vstack((data_train, data_test))
data_combined_std = np.vstack((data_train_std, data_test_std))

catagory_combined = np.hstack((catagory_train, catagory_test))

