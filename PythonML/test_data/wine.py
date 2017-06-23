# Dataset: Wine
# There are 13 different features in the Wine dataset, describing the chemical properties of the 178 wine samples.
import pandas as pd
df_wine = pd.read_csv('wine.data', header=None)
#df_wine = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data', header=None)
df_wine.columns = ['Class label', 'Alcohol', 
                   'Malic acid', 'Ash', 
                   'Alcalinity of ash', 'Magnesium',
                   'Total phenols', 'Flavanoids',
                   'Nonflavanoid phenols', 
                   'Proanthocyanins', 
                   'Color intensity', 'Hue', 
                   'OD280/OD315 of diluted wines', 
                   'Proline']
#print(df_wine.head())

from sklearn.cross_validation import train_test_split
data, catagory = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values
data_train, data_test, catagory_train, catagory_test = train_test_split(data, catagory, test_size=0.3, random_state=0)

# Standardization
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
data_train_std = stdsc.fit_transform(data_train)
data_test_std = stdsc.fit_transform(data_test)

import numpy as np
data_combined = np.vstack((data_train, data_test))
data_combined_std = np.vstack((data_train_std, data_test_std))
catagory_combined = np.hstack((catagory_train, catagory_test))

