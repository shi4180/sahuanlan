print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import shap
import sklearn
import pandas as pd
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
from sklearn.model_selection import KFold

from collections import Counter
def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    if axes is None:
        _, axes = plt.subplots(1, 1, figsize=(20, 5))

    axes.set_title(title)
    if ylim is not None:
        axes.set_ylim(*ylim)
    axes.set_xlabel("Training examples")
    axes.set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes.grid()
    axes.fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes.fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes.plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes.plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes.legend(loc="best")



    return plt


fig, axes = plt.subplots(1, 1, figsize=(10, 15))

dftrain = pd.read_csv('D:\project\sahuanlan\\TrainAllDatanew.CSV')
y_train = dftrain['Class']
X_train = dftrain.drop(['Class'], axis=1)


dftest = pd.read_csv('D:\project\sahuanlan\\TestAllDatanew.CSV')
y_test = dftest['Class']
X_test = dftest.drop(['Class'], axis=1)




params = {'eta': 0.001, 'n_estimators': 4000, 'gamma': 5, 'max_depth': 3, 'min_child_weight': 1,
                'colsample_bytree': 1, 'colsample_bylevel': 1, 'colsample_bynode': 1, 'subsample': 0.5,
                'reg_lambda': 1, 'reg_alpha': 0, 'seed': 33}
cv= KFold(n_splits=10,  random_state=0, shuffle=True)


estimator = XGBClassifier(**params)
title = "Learning Curves (XGBOOST)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.

plot_learning_curve(estimator, title, X_train, y_train, axes=axes, ylim=(0.4, 1.0),
                    cv=cv, n_jobs=4)



plt.show()