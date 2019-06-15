# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('titanic.csv')
X = dataset.iloc[:, [2,4,5,6,7,9]].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 1] = labelencoder_X.fit_transform(X[:, 1])

# Getting rid of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' , strategy = 'median', axis = 0)
imputer.fit(X[:, [2]])
X[:, [2]] = imputer.transform(X[:, [2]])



# Splitting the dataset using sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

"""
# Fitting Logistic regression to the training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)
# Predicting
y_pred = classifier.predict(X_test)

# Analysing by confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("That is 79 per cent accuracy mate!" )

# AUC SCORE
from sklearn import metrics
AUC_1 = metrics.roc_auc_score(y_test, y_pred)

"""
# Defining the Sigmoid function
def sigmoid(X, weight):
    z = np.array(np.dot(X, weight), dtype = np.float64) #Important conversion of dtype
    return 1 / (1 + np.exp(-z))

# The loss function in Logistic regression
def loss(h, y):
    return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()

# Gradient Descent
def gradient_descent(X, h, y):
    return np.dot(X.T, (h - y)) / y.shape[0]  # X dot T is used for transpose
def update_weight_loss(weight, learning_rate, gradient):
    return weight - learning_rate * gradient

# Gradient Descent in Action
num_iter = 100000
weight = np.zeros(X_train.shape[1])

for i in range(num_iter):
    h = sigmoid(X, weight)
    gradient = gradient_descent(X, h, y)
    weight = update_weight_loss(weight, 0.1, gradient)

print("Learning rate: {}\nIteration: {}".format(0.1, num_iter))

result = sigmoid(X_test, weight)

f = pd.DataFrame(np.around(result, decimals=6))
f['pred'] = f[0].apply(lambda x : 0 if x < 0.5 else 1)
print("Accuracy (Loss minimization):")
f.loc[f['pred']==y_test].shape[0] / f.shape[0] * 100

# Analysing by confusion matrix
from sklearn.metrics import confusion_matrix
cm_2 = confusion_matrix(y_test, f['pred'])

# AUC SCORE
from sklearn import metrics
AUC_2 = metrics.roc_auc_score(y_test, f['pred'])

