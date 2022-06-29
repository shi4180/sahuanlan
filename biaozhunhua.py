from sklearn import preprocessing
import numpy as np
import pandas as pd
X = pd.read_csv('D:\\project\\sahuanlan\\ICCdata.csv')
Xb = preprocessing.StandardScaler().fit(X)
Xbl = Xb.transform(X)
Xbld = pd.DataFrame(Xbl)
Xbld.to_csv("D:\\project\\sahuanlan\\normalizeICCdata.csv", header=True, index=None)