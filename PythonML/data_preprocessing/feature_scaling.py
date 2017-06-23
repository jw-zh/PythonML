import os
import sys
sys.path.append(os.getcwd())
from test_data.iris import *

# Normalization: min-max scaling.
from sklearn.preprocessing import MinMaxScaler
mms = MinMaxScaler()
data_train_norm = mms.fit_transform(data_train)
data_test_norm = mms.fit_transform(data_test)

# Standardization
from sklearn.preprocessing import StandardScaler
stdsc = StandardScaler()
data_train_std = stdsc.fit_transform(data_train)
data_test_std = stdsc.fit_transform(data_test)