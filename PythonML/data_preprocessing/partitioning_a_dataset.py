import os
import sys
sys.path.append(os.getcwd())
from test_data.wine import *

from sklearn.cross_validation import train_test_split
X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
"""
First, we assigned the NumPy array representation of feature columns 1-13 to the variable X, 
and we assigned the class labels from the first column to the variable y. 
Then, we used the train_test_split function to randomly split X and y into separate training and test datasets. 
By setting test_size=0.3 we assigned 30 percent of the wine samples to X_test and y_test, 
and the remaining 70 percent of the samples were assigned to X_train and y_train, respectively.
"""