# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
dataset = pd.read_csv('titanic.csv')
X = dataset.iloc[:, [1,2,4,6,7,9]].values
y = dataset.iloc[:, 5].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X[:, 2] = labelencoder_X.fit_transform(X[:, 2])

# Getting rid of missing data
y = y.reshape(-1,1)
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN' , strategy = 'mean', axis = 0)
imputer.fit(y)
y = imputer.transform(y)

# Splitting the dataset using sklearn
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)

"""
# Linear Regression using Sklearn
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
# Predicting the ages
y_pred = regressor.predict(X_test)

# Calculating MAPE
from sklearn.utils import check_array
def mean_absolute_percentage_error(y_test, y_pred): 
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

MAPE_1 = mean_absolute_percentage_error(y_test, y_pred)

"""
#set hyper parameters
alpha = 0.01
iters = 100000

ones = np.ones([X_train.shape[0],1])
X_train = np.concatenate((ones,X_train),axis=1)

ones = np.ones([X_test.shape[0],1])
X_test = np.concatenate((ones,X_test),axis=1)

theta = np.zeros(X_train.shape[1])

# computecost
def computeCost(X,y,theta):
    tobesummed = np.power(((X * theta.T)-y),2)
    return np.sum(tobesummed)/(2 * len(X))

#gradient descent
def gradientDescent(X,y,theta,iters,alpha):
    cost = np.zeros(iters)
    for i in range(iters):
        theta = theta - (alpha/len(X)) * np.sum(X * (X * theta.T - y), axis=0)
        cost[i] = computeCost(X, y, theta)
    
    return theta,cost

# running the gd and cost function
g, cost = gradientDescent(X_train,y_train,theta,iters,alpha)

finalCost = computeCost(X_train,y_train,g)

# plotting the cost function
fig, ax = plt.subplots()  
ax.plot(np.arange(iters), cost, 'r')  
ax.set_xlabel('Iterations')  
ax.set_ylabel('Cost')  
ax.set_title('Error vs. Training Epoch')

f = (X_test*g.T)

# Calculating MAPE
from sklearn.utils import check_array
def mean_absolute_percentage_error(y_test, y_pred): 
    return np.mean(np.abs((y_test - y_pred) / y_test)) * 100

MAPE_2 = mean_absolute_percentage_error(y_test, f)


