import numpy as np
import pandas as pd

from sklearn.ensemble import  RandomForestClassifier,AdaBoostClassifier,ExtraTreesClassifier,BaggingClassifier
from sklearn.tree import  DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score


RANDOM_STATE = 123

# Ensemble method
def ensemble_compare(X,y):
    """Compare the ensemble methods

    Args:
        X (ndarray): The data of features
        y (ndarray): the outcome of interest

    Returns:
        obb_score[dict]: the obb scores of each model
    """
    clf = DecisionTreeClassifier(random_state=123)
    # Define the classifiers
    Classifiers = (
        RandomForestClassifier(criterion='gini',random_state=123,oob_score=True),
        ExtraTreesClassifier(bootstrap=True,criterion='gini',random_state=123,oob_score=True),
        BaggingClassifier(base_estimator=clf,bootstrap=True, oob_score=True, random_state=123)
    )

    obb_score={}
    Classifiers_names = ['RandomForest','ExtraTrees','Baggaing']
    i = 0
    for classifier in Classifiers:
        classifier.fit(X,y)
        classifier_name = Classifiers_names[i]
        i+=1
        obb_score[classifier_name] = round(classifier.oob_score_,3)
    
    return obb_score

# classifier
def classifier_compare(X,y,X_test,y_test):

    clf2 = Pipeline([('scl', StandardScaler()),
                 ('clf', LogisticRegression(solver='liblinear',
                                            random_state=RANDOM_STATE))
                 ])

    clf3 = DecisionTreeClassifier(random_state=RANDOM_STATE)

    clf_labels = ['LR',  # LogisticRegression
                  'DT']  # Decision Tree
    all_clf = [clf2, clf3]
    accuracy = {}
    for clf, label in zip(all_clf, clf_labels):
        clf.fit(X, y)
        accuracy[label] = {'train': round(accuracy_score(y,clf.predict(X)),3),
                           'test':round(accuracy_score(y_test,clf.predict(X_test)),3)}
    return accuracy
