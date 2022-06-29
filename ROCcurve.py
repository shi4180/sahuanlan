import pandas as pd
from sklearn.metrics import plot_roc_curve
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import numpy as np
import shap
import sklearn
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from collections import Counter
dftrain = pd.read_csv('D:\\project\\sahuanlan\\tyTrainData.CSV')
y_train = dftrain['Class']
X_train = dftrain.drop(['Class'], axis=1)


dftest = pd.read_csv('D:\\project\\sahuanlan\\tyTestData.CSV')
y_test = dftest['Class']
X_test = dftest.drop(['Class'], axis=1)




params = {'eta': 0.001, 'n_estimators': 4000, 'gamma': 5, 'max_depth': 2, 'min_child_weight': 1,
                'colsample_bytree': 1, 'colsample_bylevel': 1, 'colsample_bynode': 1, 'subsample': 0.5,
                'reg_lambda': 1, 'reg_alpha': 0, 'seed': 33}
cv= KFold(n_splits=10,  random_state=0, shuffle=True)



for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model = XGBClassifier(**params).fit(X_train.iloc[train], y_train.iloc[train])
        predictions = model.predict(X_train.iloc[test])
        actuals = y_train.iloc[test]
        acc = sklearn.metrics.accuracy_score(actuals, predictions)
        print("Accuracy: %.2f%%" % (acc * 100.0))


for i, (train, test) in enumerate(cv.split(X_train, y_train)):
        model = XGBClassifier(**params).fit(X_train.iloc[train], y_train.iloc[train])
        predictions = model.predict(X_train.iloc[test])
        actuals = y_train.iloc[test]
        acc = sklearn.metrics.accuracy_score(actuals, predictions)
        print("Accuracy: %.2f%%" % (acc * 100.0))
xgboosttrain = plot_roc_curve(model, X_train, y_train, name="XGBClassfier in the training data set")
xgboosttest = plot_roc_curve(model, X_test, y_test, ax=xgboosttrain.ax_, name="XGBClassfier in the test data set")




plt.show()