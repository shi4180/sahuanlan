import pandas as pd
import numpy as np
from sklearn.model_selection import RepeatedKFold, cross_validate
import xgboost
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
import shap
import sklearn
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import RepeatedKFold
from sklearn.model_selection import cross_validate
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
sklearn.metrics.plot_det_curve(model, X_train, y_train, name = "training data set")
plt.show()
sklearn.metrics.plot_det_curve(model, X_test, y_test,  name = "test data set")
plt.show()


y_train=y_train.to_frame()
y_pred = model.predict(X_test)
y_predtrain = model.predict(X_train)
y_dtest = model.predict_proba(X_test)
y_dtrain = model.predict_proba(X_train)

y_dtrain = pd.DataFrame(y_dtrain)

fpr, fnr, thresholds = sklearn.metrics.det_curve(y_train, y_dtrain)
display = sklearn.metrics.DetCurveDisplay(
          fpr=fpr, fnr=fnr, estimator_name='example estimator')
display.plot()
plt.show()