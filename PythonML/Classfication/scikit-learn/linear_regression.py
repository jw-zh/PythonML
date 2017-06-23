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

# Train with logistic regression model.
from sklearn.linear_model import LogisticRegression
lr_test = LogisticRegression(C=1000.0, random_state=0)
lr_test.fit(data_train_std, catagory_train)

# Function: Plot the decision regions.
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
def plot_decision_regions(X, y, classifier, 
                    test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                         np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)
        
    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]   
        plt.scatter(X_test[:, 0], X_test[:, 1], c='', 
                alpha=1.0, linewidths=1, marker='o', 
                s=55, label='test set')
# Specify the indices of the samples that we want to mark on the resulting plots.
data_combined_std = np.vstack((data_train_std, data_test_std))
catagory_combined = np.hstack((catagory_train, catagory_test))
plot_decision_regions(X=data_combined_std, y=catagory_combined, classifier=lr_test, test_idx=range(105,150))
plt.xlabel('petal length [standardized]') 
plt.ylabel('petal width [standardized]') 
plt.legend(loc='upper left')
plt.show()


weights, params = [], []
for c in np.arange(-5, 5):
    lr = LogisticRegression(C=10**c, random_state=0)
    lr.fit(data_train_std, catagory_train)
    weights.append(lr.coef_[1])
    params.append(10**c)
weights = np.array(weights)
plt.plot(params, weights[:, 0], 
         label='petal length')
plt.plot(params, weights[:, 1], linestyle='--', 
         label='petal width')
plt.ylabel('weight coefficient')
plt.xlabel('C')
plt.legend(loc='upper left')
plt.xscale('log')
plt.show()
"""
By executing the preceding code, we fitted ten logistic regression models with different values for the inverse-regularization parameter C. 
For the purposes of illustration, we only collected the weight coefficients of the class 2 vs. all classifier. 
Remember that we are using the OvR technique for multiclass classification.

As we can see in the resulting plot, 
the weight coefficients shrink if we decrease the parameter C, 
that is, if we increase the regularization strength:
"""