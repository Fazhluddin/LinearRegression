import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model

# Importing the dataset
dataset = pd.read_csv("RetailData.csv")

# Ireating the array of independent variables
X = dataset.iloc[:, [7, 8, 12]].values

# Dependent variable(Total Amount)
Y = dataset.iloc[:, [13]].values

X_train, X_test, Y_train, Y_test = train_test_split(
                                                    X, Y, test_size=0.1,
                                                    random_state=1)

# Linear regression
# Assigning linear regression model to a variable
model = linear_model.LinearRegression()

# Fitting independent variables x_train and dependent variable y_train to model
model.fit(X_train, Y_train)

# Dumping the model
pickle.dump(model, open("model.pkl", "wb"))
