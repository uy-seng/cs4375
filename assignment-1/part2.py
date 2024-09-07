# %%
import pandas as pd
import numpy as np
import requests
from io import StringIO
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
import logging

logging.basicConfig(filename='output_part2.log', level=logging.INFO, format='%(asctime)s - %(message)s')

# %%
def async_read_csv(url):
    response = requests.get(url, stream=True)
    response.raise_for_status() # will raise error if request is not successful
    return pd.read_csv(StringIO(response.text))
    
# Read the streamed content
df = async_read_csv("https://raw.githubusercontent.com/uy-seng/cs4375/main/assignment-1/scripts/convert_to_csv/abalone.csv")
df.head(), df.shape

# %%
df.columns

# %%
df = df.replace({"M":1, "F": 2, "I": 3})

# %%
df.corr()["rings"]

# %%
features = ['sex', 'length', 'diameter', 'height', 'whole_weight', 'shucked_weight',
       'viscera_weight', 'shell_weight']

x = df[features]
y = df["rings"]

# %%
x_train, x_test, y_train, y_test =  train_test_split(x, y, test_size=0.2)

# %%
from sklearn.linear_model import SGDRegressor

sgd = SGDRegressor()

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import make_scorer, r2_score

# Define the parameter grid
param_grid = {
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate': ['constant', 'optimal', 'invscaling', 'adaptive'],
    'max_iter': [1000, 2000, 3000],
    'tol': [1e-3, 1e-4, 1e-5]
}

# Initialize the SGDRegressor
sgd = SGDRegressor()

# Set up the GridSearchCV with 5-fold cross-validation
grid_search = GridSearchCV(estimator=sgd, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2, scoring='r2')

# Fit the grid search model
grid_search.fit(x_train, y_train)

# Print R² score for each parameter combination
print("R² scores for each parameter combination:")
results = pd.DataFrame(grid_search.cv_results_)
for i in range(len(results)):
    params = results['params'][i]
    r2_mean = results['mean_test_score'][i]
    r2_std = results['std_test_score'][i]
    logging.info(f"Parameters: {params} => R² mean: {r2_mean:.4f}, R² std: {r2_std:.4f}")

# Print the best parameters and best score
print(f"\nBest parameters found: {grid_search.best_params_}")
print(f"Best R² score: {grid_search.best_score_}")

# Evaluate the best model on the test set
best_model = grid_search.best_estimator_
y_pred = best_model.predict(x_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
ev = explained_variance_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"\nTest set performance:")
print(f"Mean Squared Error: {mse}")
print(f"Mean Absolute Error: {mae}")
print(f"Explained Variance: {ev}")
print(f"R² Score: {r2}") 

# # %%
# sgd.fit(x_train, y_train)

# # %%
# sgd.score(x_test, y_test)

# # %%
# y_pred = sgd.predict(x_test)
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# ev = explained_variance_score(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# # %%
# mse, mae, ev, r2


