import pandas as pd
from sklearn.model_selection import train_test_split
df = pd.read_csv('D:\\project\\sahuanlan\\normalizeICCdata.csv')
y = df['Class']
X = df.drop(['Class'], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=2)

X_train.to_csv("D:\\project\\sahuanlan\\X_train.csv", header=True, index=None)
y_train.to_csv("D:\\project\\sahuanlan\\y_train.csv", header=True, index=None)
X_test.to_csv("D:\\project\\sahuanlan\\X_test.csv", header=True, index=None)
y_test.to_csv("D:\\project\\sahuanlan\\y_test.csv", header=True, index=None)
