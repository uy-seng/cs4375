'''
Problem Statement:
You are a data scientist working for a real estate company. Your task is to predict the price of houses in a given neighborhood based on various features of the houses. The company has provided you with a dataset containing historical data on house prices and features.

Dataset:
The dataset contains the following features:

Square Footage (X1): The total square footage of the house.
Number of Bedrooms (X2): The number of bedrooms in the house.
Number of Bathrooms (X3): The number of bathrooms in the house.
(X4): The age of the house in years.
Distance to City Center (X5): The distance of the house from the city center in miles.

The target variable is the Price of the House (Y).


Task:
- Implement multivariate linear regression using gradient descent to predict house prices based on the features provided.
- Initialize the parameters theta0 (bias) and theta1, theta2 ... theta5
- Use a learning rate 𝛼 of your choice, but typically start with something like 0.01.
- Run the gradient descent algorithm for a sufficient number of iterations (e.g., 1000 iterations) to ensure convergence.
- Plot the cost function (MSE) over iterations to visualize the convergence.
- Output the final values of theta0, theta1 ... theta5.
- Evaluate the performance of your model using metrics such as Mean Absolute Error (MAE) or R-squared.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from enum import Enum
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


class BooleanChoice(Enum):
    YES = 0
    NO = 1
    
class FurnishStatus(Enum):
    UNFURNISHED = 0
    SEMI_FURNISHED = 1
    FURNISHED = 2
   

class GradientDescent:
    def __init__(self, learning_rate=0.01, iterations=1000):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.theta = None
        self.cost_history = []

    def compute_cost(self, X, y):
        m = len(y)
        predictions = np.dot(X, self.theta)
        cost = (1/2*m) * np.sum(np.square(predictions - y))
        return cost

    def fit(self, X, y):
        m = len(y)
        self.theta = np.zeros(X.shape[1])

        for i in range(self.iterations):
            predictions = np.dot(X, self.theta)
            errors = predictions - y
            self.theta -= (self.learning_rate / m) * np.dot(X.T, errors)
            cost = self.compute_cost(X, y)
            self.cost_history.append(cost)

        return self.theta

'''
Read in from csv files
'''
df = pd.read_csv("./data.csv")
# change all yes and no to enum BooleanChoice[value.toupper()]
for col in df.columns:
    if set(df[col].unique()) ==  {"yes", "no"}:
        df[col] = df[col].apply(lambda x: BooleanChoice[x.upper()].value)

df["furnishingstatus"] = df["furnishingstatus"].apply(lambda x: FurnishStatus[x.upper().replace("-", "_")].value)

X = df.drop("price", axis=1).values
y = df["price"].values

# Split into training and testing sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Adding a column of ones to X_train and X_test for the bias term
X_train = np.c_[np.ones(X_train.shape[0]), X_train]
X_test = np.c_[np.ones(X_test.shape[0]), X_test]


# Initialize and fit the model
gd = GradientDescent(learning_rate=0.01, iterations=1000)
theta = gd.fit(X_train, y_train)


# Plot the cost function over iterations to visualize convergence
# plt.plot(gd.cost_history)
# plt.xlabel("Iteration")
# plt.ylabel("Cost (MSE)")
# plt.title("Cost Function Over Iterations")
# plt.show()

# Make predictions on the test set
predictions = np.dot(X_test, theta)

# Calculate R-squared
ss_res = np.sum((y_test - predictions) ** 2)
ss_tot = np.sum((y_test - np.mean(y_test)) ** 2)
r_squared = 1 - (ss_res / ss_tot)

print(f"R-squared: {r_squared}")
                
            