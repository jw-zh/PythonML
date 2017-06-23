# Generate sample data.
import pandas as pd
from io import StringIO
csv_data = '''A,B,C,D 
1.0,2.0,3.0,4.0 
5.0,6.0,,8.0
10.0,11.0,12.0,'''
df = pd.read_csv(StringIO(csv_data))
print(df)
# Simply drop data.
print(df.isnull().sum())
print(df.values)
print(df.dropna())
print(df.dropna(axis=1))
# only drop rows where all columns are NaN
df.dropna(how='all')  
# drop rows that have not at least 4 non-NaN values
df.dropna(thresh=4)  
# only drop rows where NaN appear in specific columns (here: 'C')
df.dropna(subset=['C'])

# Interpolation techniques
# 1. Mean imputation: simply replace the missing value by the mean value of the entire feature column.
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN', strategy='mean', axis=0)
imputer = imputer.fit(df)
imputed_data = imputer.transform(df.values)
print(imputed_data)

# Sample data #2.
df = pd.DataFrame([
    ['green', 'M', 10.1, 'class1'], 
    ['red', 'L', 13.5, 'class2'], 
    ['blue', 'XL', 15.3, 'class1']])
df.columns = ['color', 'size', 'price', 'classlabel']
print(df)
size_mapping = {'XL': 3,
                'L': 2,
                'M': 1}
df['size'] = df['size'].map(size_mapping) # transform string values to integers
print(df)
# Encoding class labels, simply enumerate the class labels starting at 0
import numpy as np
class_mapping = {label:idx for idx, label in enumerate(np.unique(df['classlabel']))}
print(class_mapping)
df['classlabel'] = df['classlabel'].map(class_mapping)
print(df)
inv_class_mapping = {v:k for k, v in class_mapping.items()}
df['classlabel'] = df['classlabel'].map(inv_class_mapping)
print(df)
# Alternatively, use LabelEncoder to achieve the same
from sklearn.preprocessing import LabelEncoder
class_le = LabelEncoder()
y = class_le.fit_transform(df['classlabel'].values)
print(y)
y = class_le.inverse_transform(y)
print(y)
# LabelEncoder on nominal features
X = df[['color', 'size', 'price']].values
color_le = LabelEncoder()
X[:, 0] = color_le.fit_transform(X[:, 0])
print(X)
# One-hot encoding
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(categorical_features=[0])
print(ohe.fit_transform(X).toarray())
# A more convenient way
pd.get_dummies(df[['price', 'color', 'size']])