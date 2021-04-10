# --------------------------< section 1: Load neceaasy packages>------------------------
# base
import time
# Data
import numpy as np
import pandas as pd

# Plot
import seaborn as sns
import matplotlib.pyplot as plt

# machine learning
from sklearn import preprocessing
from sklearn import svm
from sklearn import model_selection as ms
from sklearn.pipeline import make_pipeline

hotel = pd.read_csv(r'Code\Data\hotel_pre.csv')
# --------------------------< section 2: data processing>--------------------------------
# data-spliting: train and test
y = hotel['is_canceled']
X = hotel.drop(['is_canceled'], axis=1)
X_train, X_test, y_train, y_test = ms.train_test_split(X, y, test_size=0.2)
# --------------------------< section 3: Create the model>-------------------------------
t0 = time.time()
param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],}
clf = svm.SVC(kernel='linear', class_weight='balanced')
clf = clf.fit(X_train, y_train)
print(clf.best_estimator_)
print()
# --------------------------< section 4: model tuning>-----------------------------------


# --------------------------< section 5: model check>------------------------------------
